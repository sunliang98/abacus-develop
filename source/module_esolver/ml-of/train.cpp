#include "./train.h"
#include <sstream>
#include <math.h>
#include "time.h"

Train::~Train()
{
    delete[] this->train_dir;
    delete[] this->train_cell;
    delete[] this->train_a;
    delete[] this->train_volume;
    delete[] this->validation_dir;
    delete[] this->validation_cell;
    delete[] this->validation_a;
    delete[] this->vali_volume;
}

void Train::init()
{
    if (this->loss == "potential" || this->loss == "both" || this->loss == "both_new") this->setUpFFT();
    this->nn = std::make_shared<NN_OFImpl>(this->nx_train, this->ninput);
    // this->nn->to(device);
    this->nn->setData(this->nn_input_index,
                      this->gamma.reshape({this->nx_train}),
                      this->p.reshape({this->nx_train}), 
                      this->q.reshape({this->nx_train}), 
                      this->gammanl.reshape({this->nx_train}), 
                      this->pnl.reshape({this->nx_train}), 
                      this->qnl.reshape({this->nx_train}),
                      this->xi.reshape({this->nx_train}),
                      this->tanhxi.reshape({this->nx_train}),
                      this->tanhxi_nl.reshape({this->nx_train}),
                      this->tanhp.reshape({this->nx_train}),
                      this->tanhq.reshape({this->nx_train}),
                      this->tanh_pnl.reshape({this->nx_train}),
                      this->tanh_qnl.reshape({this->nx_train}),
                      this->tanhp_nl.reshape({this->nx_train}),
                      this->tanhq_nl.reshape({this->nx_train}));
    // if (this->loss == "potential" || this->loss == "both" || this->loss == "both_new") this->nn->inputs.requires_grad_(true);
}

torch::Tensor Train::lossFunction(torch::Tensor enhancement, torch::Tensor target, torch::Tensor coef)
{
    return torch::sum(torch::pow(enhancement - target, 2))/this->nx/coef/coef;
}

torch::Tensor Train::lossFunction_new(torch::Tensor enhancement, torch::Tensor target, torch::Tensor tauTF, torch::Tensor coef)
{
    return torch::sum(torch::pow(tauTF * (enhancement - target), 2.))/this->nx/coef/coef;
}


void Train::train()
{
    // time
    double tot = 0.;
    double totFback = 0.;
    double totLoss = 0.;
    double totLback = 0.;
    time_t start, startFB, startL, startLB, end, endFB, endL, endLB;

    start = clock();

    std::cout << "========== Train begin ==========" << std::endl;
    torch::Tensor target = (this->loss=="energy") ? this->enhancement : this->pauli;
    auto dataset = OF_data(this->nn->inputs, target).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(dataset, this->nbatch);
    if (this->loss == "potential" || this->loss == "both" || this->loss == "both_new")
    {
        this->pauli.resize_({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    }

    torch::optim::SGD optimizer(this->nn->parameters(), this->lr_start);
    double update_coef = this->nepoch/std::log(this->lr_end/this->lr_start); // used to reduce the learning rate

    std::cout << "Epoch\tLoss\tValidation\tLoss_pot\tLoss_E\tLoss_FEG_pot\tLoss_FEG_E\n";
    double lossTrain = 0.;
    double lossPot = 0.;
    double lossE = 0.;
    double lossFEG_pot = 0.;
    double lossFEG_E = 0.;
    double lossVali = 0.;
    double maxLoss = 100.;

    // bool increase_coef_feg_e = false;
    torch::Tensor tauTF = this->cTF * torch::pow(this->rho, 5./3.);
    for (size_t epoch = 1; epoch <= this->nepoch; ++epoch)
    {
        size_t batch_index = 0;
        for (auto& batch : *data_loader)
        {
            optimizer.zero_grad();
            if (this->loss == "energy")
            {
                torch::Tensor prediction = this->nn->forward(batch.data);
                startL = clock();
                torch::Tensor loss = this->lossFunction(prediction, batch.target) * this->coef_e;
                lossTrain = loss.item<double>();
                endL = clock();
                totLoss += endL - startL;

                startLB = clock();
                loss.backward();
                endLB = clock();
                totLback += endLB - startLB;
            }
            else if (this->loss == "potential" || this->loss == "both" || this->loss == "both_new")
            {
                torch::Tensor inpt = torch::slice(this->nn->inputs, 0, batch_index*this->nx, (batch_index + 1)*this->nx);
                inpt.requires_grad_(true);
                torch::Tensor prediction = this->nn->forward(inpt);
                if (this->feg_limit != 0)
                {
                    // if (this->ml_gamma) if (this->feg_inpt[this->nn_input_index["gamma"]].grad().numel()) this->feg_inpt[this->nn_input_index["gamma"]].grad().zero_();
                    if (this->feg_inpt.grad().numel()) this->feg_inpt.grad().zero_();
                    this->feg_predict = this->nn->forward(this->feg_inpt);
                    // if (this->ml_gamma) this->feg_dFdgamma = torch::autograd::grad({this->feg_predict}, {this->feg_inpt[this->nn_input_index["gamma"]]},
                    //                                                                 {torch::ones(1)}, true, true)[0];
                    if (this->ml_gamma) this->feg_dFdgamma = torch::autograd::grad({this->feg_predict}, {this->feg_inpt},
                                                                                    {torch::ones_like(this->feg_predict)}, true, true)[0][this->nn_input_index["gamma"]];
                    if (this->feg_limit == 1) prediction = prediction - this->feg_predict + 1.;
                }
                startFB = clock();
                torch::Tensor gradient = torch::autograd::grad({prediction}, {inpt},
                                                           {torch::ones_like(prediction)}, true, true)[0];
                // std::cout << gradient.requires_grad() << std::endl;
                // gradient.requires_grad_(true);
                endFB = clock();
                totFback += endFB - startFB;

                startL = clock();
                torch::Tensor pot = this->getPot(
                    this->rho[batch_index],
                    this->nablaRho[batch_index],
                    tauTF[batch_index],
                    this->gamma[batch_index],
                    this->p[batch_index],
                    this->q[batch_index],
                    this->xi[batch_index],
                    this->tanhxi[batch_index],
                    this->tanhxi_nl[batch_index],
                    this->tanhp[batch_index],
                    this->tanhq[batch_index],
                    this->tanh_pnl[batch_index],
                    this->tanh_qnl[batch_index],
                    prediction.reshape({this->fftdim, this->fftdim, this->fftdim}),
                    gradient,
                    this->fft_kernel_train[batch_index],
                    this->fft_grid_train[batch_index],
                    this->fft_gg_train[batch_index]
                );
                torch::Tensor loss = this->lossFunction(pot, this->pauli[batch_index], this->pauli_mean[batch_index])
                                     * this->coef_p;
                lossPot = loss.item<double>();
                if (this->loss == "both")
                {
                    loss = loss + this->coef_e * this->lossFunction(prediction, torch::slice(this->enhancement, 0, batch_index*this->nx, (batch_index + 1)*this->nx), this->enhancement_mean[batch_index]);
                    lossE = loss.item<double>() - lossPot;
                }
                if (this->loss == "both_new")
                {
                    loss = loss + this->coef_e * this->lossFunction_new(prediction, torch::slice(this->enhancement, 0, batch_index*this->nx, (batch_index + 1)*this->nx), tauTF[batch_index].reshape({this->nx, 1}), this->tau_mean[batch_index]);
                    lossE = loss.item<double>() - lossPot;
                }
                if (this->feg_limit != 0)
                {
                    loss = loss + torch::pow(this->feg_dFdgamma, 2) * this->coef_feg_p;
                    lossFEG_pot = loss.item<double>() - (lossPot + lossE);
                    if (this->feg_limit == 2)
                    {
                        loss = loss + torch::pow(this->feg_predict - 1., 2) * this->coef_feg_e;
                        lossFEG_E = loss.item<double>() - (lossPot + lossE + lossFEG_pot);
                        // if (lossFEG_E/lossE < 1e-3 && increase_coef_feg_e == false)
                        // {
                        //     this->coef_feg_e *= 2.;
                        //     increase_coef_feg_e = true;
                        //     std::cout << "---------ICREASE COEF FEG E--------" << std::endl;
                        // }
                    }
                }

                lossTrain = loss.item<double>();
                endL = clock();
                totLoss += endL - startL;
                
                startLB = clock();
                loss.backward();
                endLB = clock();
                totLback += endLB - startLB;
                // this->dumpTensor(pot.reshape({this->nx}), "pot_fcc.npy", this->nx);
                // this->dumpTensor(torch::slice(prediction, 0, batch_index*this->nx, (batch_index + 1)*this->nx).reshape({this->nx}), "F_fcc.npy", this->nx);
            }
            
            if (epoch % this->print_fre == 0) {
                if (this->nvalidation > 0)
                {
                    torch::Tensor valid_pre = this->nn->forward(this->input_vali);
                    lossVali = this->lossFunction(valid_pre, this->enhancement_vali, this->enhancement_mean_vali).item<double>();
                }
                std::cout << std::setiosflags(std::ios::scientific) << std::setprecision(3) << epoch 
                          << std::setw(12) << lossTrain 
                          << std::setw(12) << lossVali 
                          << std::setw(12) << lossPot
                          << std::setw(12) << lossE 
                          << std::setw(12) << lossFEG_pot
                          << std::setw(12) << lossFEG_E
                          << std::endl;
            }
            
            if (lossTrain > maxLoss){
                std::cout << "ERROR: too large loss: " << lossTrain << std::endl;
                exit(0);
            }
            optimizer.step();

            batch_index += 1;
        }
        if (epoch % this->dump_fre == 0)
        {
            std::stringstream file;
            file << "model/net" << epoch << ".pt";
            torch::save(this->nn, file.str());
        }
        // Reduce the learning_rate
        if (epoch % this->lr_fre == 0)
        {
            for (auto &group : optimizer.param_groups())
            {
                if(group.has_options())
                {
                    auto &options = static_cast<torch::optim::SGDOptions &>(group.options());
                    // options.lr(this->step_length * std::exp(epoch/30.));
                    options.lr(this->lr_start * std::exp(epoch/update_coef));
                }
            }
        }
    }
    end = clock();
    tot = end - start;
    std::cout << "=========== Done ============" << std::endl;
    std::cout << std::setprecision(1) << "Item\t\t\tTime (s)\tPercentage (%)" << std::endl;
    std::cout << "Total\t\t\t" << tot/1000000. << "\t\t" << tot/tot << std::endl;
    std::cout << "Enhancement back\t" << totFback/1000000. << "\t\t" << totFback/tot << std::endl;
    std::cout << "Loss function\t\t" << totLoss/1000000. << "\t\t" << totLoss/tot << std::endl;
    std::cout << "Loss backward\t\t" << totLback/1000000. << "\t\t" << totLback/tot << std::endl;
}

void Train::potTest()
{
    time_t start, end;
    if (this->loss == "potential" || this->loss == "both" || this->loss == "both_new") this->setUpFFT();
    this->nn = std::make_shared<NN_OFImpl>(this->nx_train, this->ninput);
    torch::load(this->nn, "net.pt");

    this->nn->setData(this->nn_input_index,
                      this->gamma.reshape({this->nx_train}),
                      this->p.reshape({this->nx_train}), 
                      this->q.reshape({this->nx_train}), 
                      this->gammanl.reshape({this->nx_train}), 
                      this->pnl.reshape({this->nx_train}), 
                      this->qnl.reshape({this->nx_train}),
                      this->xi.reshape({this->nx_train}),
                      this->tanhxi.reshape({this->nx_train}),
                      this->tanhxi_nl.reshape({this->nx_train}),
                      this->tanhp.reshape({this->nx_train}),
                      this->tanhq.reshape({this->nx_train}),
                      this->tanh_pnl.reshape({this->nx_train}),
                      this->tanh_qnl.reshape({this->nx_train}),
                      this->tanhp_nl.reshape({this->nx_train}),
                      this->tanhq_nl.reshape({this->nx_train}));
    this->nn->inputs.requires_grad_(true);

    torch::Tensor target = (this->loss=="energy") ? this->enhancement : this->pauli;
    auto dataset = OF_data(this->nn->inputs, target).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(dataset, this->nbatch);
    if (this->loss == "potential" || this->loss == "both" || this->loss == "both_new") this->pauli.resize_({this->ntrain, this->fftdim, this->fftdim, this->fftdim});

    for (auto& batch : *data_loader)
    {
        for (int ii = 0; ii < 1; ++ii)
        {
            torch::Tensor inpts = torch::slice(this->nn->inputs, 0, ii*this->nx, (ii + 1)*this->nx);
            inpts.requires_grad_(true);
            torch::Tensor prediction = this->nn->forward(inpts);

            if (this->feg_limit != 0)
            {
                // if (this->ml_gamma) if (this->feg_inpt[this->nn_input_index["gamma"]].grad().numel()) this->feg_inpt[this->nn_input_index["gamma"]].grad().zero_();
                if (this->feg_inpt.grad().numel()) this->feg_inpt.grad().zero_();
                this->feg_predict = this->nn->forward(this->feg_inpt);
                // if (this->ml_gamma) this->feg_dFdgamma = torch::autograd::grad({this->feg_predict}, {this->feg_inpt[this->nn_input_index["gamma"]]},
                //                                                                 {torch::ones(1)}, true, true)[0];
                if (this->ml_gamma) this->feg_dFdgamma = torch::autograd::grad({this->feg_predict}, {this->feg_inpt},
                                                                                {torch::ones_like(this->feg_predict)}, true, true)[0][this->nn_input_index["gamma"]];
                if (this->feg_limit == 1) prediction = prediction - this->feg_predict + 1.;
            }

            start = clock();
            torch::Tensor gradient = torch::autograd::grad({prediction}, {inpts},
                                                    {torch::ones_like(prediction)}, true, true)[0];
            end = clock();
            std::cout << "spend " << (end-start)/1e6 << " s" << std::endl;
            
            std::cout << "begin potential" << std::endl;
            torch::Tensor tauTF = this->cTF * torch::pow(rho, 5./3.);

            torch::Tensor pot = this->getPot(
                rho[ii],
                nablaRho[ii],
                tauTF[ii],
                gamma[ii],
                p[ii],
                q[ii],
                xi[ii],
                tanhxi[ii],
                tanhxi_nl[ii],
                tanhp[ii],
                tanhq[ii],
                tanh_pnl[ii],
                tanh_qnl[ii],
                torch::slice(prediction, 0, ii*this->nx, (ii + 1)*this->nx).reshape({this->fftdim, this->fftdim, this->fftdim}),
                gradient,
                this->fft_kernel_train[ii],
                this->fft_grid_train[ii],
                this->fft_gg_train[ii]
            );
            torch::Tensor loss = this->lossFunction(pot, this->pauli[ii]) * this->coef_p;
            if (this->loss == "both")
            {
                loss = loss + this->coef_e * this->lossFunction(prediction, torch::slice(this->enhancement, 0, ii*this->nx, (ii + 1)*this->nx), this->enhancement_mean[ii]);
            }
            if (this->loss == "both_new")
            {
                loss = loss + this->coef_e * this->lossFunction_new(prediction, torch::slice(this->enhancement, 0, ii*this->nx, (ii + 1)*this->nx), tauTF[ii].reshape({this->nx, 1}), this->tau_mean[ii]);
            }
            // loss = loss + this->coef_e * this->lossFunction(prediction, torch::slice(this->enhancement, 0, ii*this->nx, (ii + 1)*this->nx));
            double lossTrain = loss.item<double>();
            std::cout << "loss = " << lossTrain << std::endl;
            this->dumpTensor(pot.reshape({this->nx}), "potential-nnof.npy", this->nx);
            this->dumpTensor(torch::slice(prediction, 0, ii*this->nx, (ii + 1)*this->nx).reshape({this->nx}), "enhancement-nnof.npy", this->nx);
            // this->dumpTensor(torch::slice(this->nn->F, 0, ii*this->nx, (ii + 1)*this->nx).reshape({this->nx}), "F_fcc.npy", this->nx);

        }
        // std::cout << std::setiosflags(std::ios::scientific) <<  std::setprecision(12) << this->nn->parameters() << std::endl;

    }
    // // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;

    // this->nn->zero_grad();
    // this->input_vali.requires_grad_(true);
    // this->nn->F = this->nn->forward(this->input_vali);
    // if (this->input_vali.grad().numel()) this->input_vali.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.
    // // cout << "begin backward" << endl;
    // this->nn->F.backward(torch::ones({this->nx_vali, 1}));
    // // cout << this->nn->inputs.grad();
    // this->nn->gradient = this->input_vali.grad();
    // torch::Tensor pot_vali = this->getPot(
    //     rho_vali[0],
    //     nablaRho_vali[0],
    //     this->cTF * torch::pow(rho_vali[0], 5./3.),
    //     gamma_vali[0],
    //     p_vali[0],
    //     q_vali[0],
    //     this->nn->F.reshape({this->fftdim, this->fftdim, this->fftdim}),
    //     this->nn->gradient,
    //     this->fft_kernel_vali[0],
    //     this->fft_grid_vali[0],
    //     this->fft_gg_vali[0]
    // );
    // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;
    // this->dumpTensor(pot_vali.reshape({this->nx_vali}), "pot_bcc.npy", this->nx_vali);
    // this->dumpTensor(this->nn->F.reshape({this->nx_vali}), "F_bcc.npy", this->nx_vali);
}

