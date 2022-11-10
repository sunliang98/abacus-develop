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
    if (this->nvalidation > 0) this->setUpFFT();
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

int main()
{
    // train test
    Train train;
    train.readInput();
    train.loadData();
    train.initNN();
    train.train();

    // load test
    // std::shared_ptr<NN_OFImpl> nn = std::make_shared<NN_OFImpl>(64000, 6);
    // torch::load(nn, "./net.pt");
    // nn->setData(train.gamma, train.gammanl, train.p, train.pnl, train.q, train.qnl);
    // std::cout << nn->forward(nn->inputs);
}