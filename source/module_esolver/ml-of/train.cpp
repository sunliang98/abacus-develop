#include "./train.h"
#include <sstream>
#include <math.h>


Train::~Train()
{
    delete[] this->train_cell;
    delete[] this->train_a;
    delete[] this->train_volume;
    delete[] this->validation_cell;
    delete[] this->validation_a;
    delete[] this->vali_volume;
}

void Train::initNN()
{
    if (this->loss == "potential") this->setUpFFT();
    this->nn = std::make_shared<NN_OFImpl>(this->nx_train, this->ninput);
    this->nn->setData(this->nn_input_index, this->gamma, this->gammanl, this->p, this->pnl, this->q, this->qnl);
}

torch::Tensor Train::lossFunction(torch::Tensor enhancement, torch::Tensor target)
{
    return torch::sum(torch::pow(enhancement - target, 2))/this->nbatch;
}

void Train::train()
{
    std::cout << "Train begin" << std::endl;
    auto dataset = OF_data(this->nn->inputs, this->enhancement).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(dataset, this->nbatch);

    torch::optim::SGD optimizer(this->nn->parameters(), this->step_length);
    std::cout << "Epoch\tLoss\tValidation\n";
    double lossTrain = 0.;
    double lossVali = 0.;

    for (size_t epoch = 1; epoch <= this->nepoch; ++epoch)
    {
        size_t batch_index = 0;
        for (auto& batch : *data_loader)
        {
            optimizer.zero_grad();
            torch::Tensor prediction = this->nn->forward(batch.data);
            torch::Tensor loss = this->lossFunction(prediction, batch.target);
            lossTrain = loss.item<double>();
            loss.backward();
            optimizer.step();
        }
        if (epoch % this->dump_fre == 0)
        {
            std::stringstream file;
            file << "model/net" << epoch << ".pt";
            torch::save(this->nn, file.str());
        }
        if (epoch % this->print_fre == 0) {
            if (this->nvalidation > 0)
            {
                torch::Tensor valid_pre = this->nn->forward(this->input_vali);
                lossVali = this->lossFunction(valid_pre, this->enhancement_vali).item<double>();
            }
            std::cout << std::setiosflags(std::ios::scientific) << std::setprecision(3) << epoch << std::setw(12) << lossTrain <<  std::setw(12) << lossVali << std::endl;
        }
        // break;
    }
}

void Train::potTest()
{
    if (this->loss == "potential") this->setUpFFT();
    this->nn = std::make_shared<NN_OFImpl>(this->nx_train, this->ninput);
    torch::load(this->nn, "net.pt");
    this->nn->setData(this->nn_input_index, this->gamma, this->gammanl, this->p, this->pnl, this->q, this->qnl);

    this->nn->inputs.requires_grad_(true);
    this->nn->F = this->nn->forward(this->nn->inputs);
    if (this->nn->inputs.grad().numel()) this->nn->inputs.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.
    // cout << "begin backward" << endl;
    this->nn->F.backward(torch::ones({this->nx_train, 1}));
    // cout << this->nn->inputs.grad();
    this->nn->gradient = this->nn->inputs.grad();
    // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;
    std::cout << "begin potential" << std::endl;
    // std::cout << "nablaRho size" << gamma.reshape({this->fftdim, this->fftdim, this->fftdim}) << std::endl;
    torch::Tensor pot = this->getPot(
        rho.reshape({this->fftdim, this->fftdim, this->fftdim}),
        nablaRho.reshape({3, this->fftdim, this->fftdim, this->fftdim}),
        this->cTF * torch::pow(rho.reshape({this->fftdim, this->fftdim, this->fftdim}), 5./3.),
        gamma.reshape({this->fftdim, this->fftdim, this->fftdim}),
        p.reshape({this->fftdim, this->fftdim, this->fftdim}),
        q.reshape({this->fftdim, this->fftdim, this->fftdim}),
        this->nn->F.reshape({this->fftdim, this->fftdim, this->fftdim}),
        this->nn->gradient,
        this->fft_kernel_train[0],
        this->fft_grid_train[0],
        this->fft_gg_train[0]
    );

    this->dumpTensor(pot.reshape({this->nx_train}), "pot_bcc.npy", this->nx_train);
    this->dumpTensor(this->nn->F.reshape({this->nx_train}), "F_bcc.npy", this->nx_train);
    // std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;

    this->nn->zero_grad();
    this->input_vali.requires_grad_(true);
    this->nn->F = this->nn->forward(this->input_vali);
    if (this->input_vali.grad().numel()) this->input_vali.grad().zero_(); // In the first step, inputs.grad() returns an undefined Tensor, so that numel() = 0.
    // cout << "begin backward" << endl;
    this->nn->F.backward(torch::ones({this->nx_vali, 1}));
    // cout << this->nn->inputs.grad();
    this->nn->gradient = this->input_vali.grad();
    torch::Tensor pot_vali = this->getPot(
        rho_vali.reshape({this->fftdim, this->fftdim, this->fftdim}),
        nablaRho_vali.reshape({3, this->fftdim, this->fftdim, this->fftdim}),
        this->cTF * torch::pow(rho_vali.reshape({this->fftdim, this->fftdim, this->fftdim}), 5./3.),
        gamma_vali.reshape({this->fftdim, this->fftdim, this->fftdim}),
        p_vali.reshape({this->fftdim, this->fftdim, this->fftdim}),
        q_vali.reshape({this->fftdim, this->fftdim, this->fftdim}),
        this->nn->F.reshape({this->fftdim, this->fftdim, this->fftdim}),
        this->nn->gradient,
        this->fft_kernel_vali[0],
        this->fft_grid_vali[0],
        this->fft_gg_vali[0]
    );
    std::cout << torch::slice(this->nn->gradient, 0, 0, 10) << std::endl;
    this->dumpTensor(pot_vali.reshape({this->nx_vali}), "pot_fcc.npy", this->nx_vali);
    this->dumpTensor(this->nn->F.reshape({this->nx_vali}), "F_fcc.npy", this->nx_vali);
}

int main()
{
    // train test
    Train train;
    train.readInput();
    train.loadData();
    // train.initNN();
    // train.train();
    train.potTest();

    // load test
    // std::shared_ptr<NN_OFImpl> nn = std::make_shared<NN_OFImpl>(64000, 6);
    // torch::load(nn, "./net.pt");
    // nn->setData(train.gamma, train.gammanl, train.p, train.pnl, train.q, train.qnl);
    // std::cout << nn->forward(nn->inputs);
}