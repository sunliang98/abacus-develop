#include "./train.h"

void Train::setPara(int nrxx, int ninpt, int nbatch)
{
    this->nrxx = nrxx;
    this->ninpt = ninpt;
    this->nbatch = nbatch;
}

void Train::loadData()
{
    std::vector<long unsigned int> cshape = {(long unsigned) this->nrxx};
    bool fortran_order = false;
    npy::LoadArrayFromNumpy("gamma.npy", cshape, fortran_order, this->gamma);
    npy::LoadArrayFromNumpy("gammanl.npy", cshape, fortran_order, this->gammanl);
    npy::LoadArrayFromNumpy("p.npy", cshape, fortran_order, this->p);
    npy::LoadArrayFromNumpy("pnl.npy", cshape, fortran_order, this->pnl);
    npy::LoadArrayFromNumpy("q.npy", cshape, fortran_order, this->q);
    npy::LoadArrayFromNumpy("qnl.npy", cshape, fortran_order, this->qnl);

    std::vector<double> enhancement_;
    npy::LoadArrayFromNumpy("enhancement.npy", cshape, fortran_order, enhancement_);
    this->enhancement = torch::tensor(enhancement_);
    this->enhancement.resize_({this->nrxx, 1});
    std::cout << "load data done" << std::endl;
}

void Train::initNN()
{
    this->nn.setPara(this->nrxx, this->ninpt);
    this->nn.setData(this->gamma, this->gammanl, this->p, this->pnl, this->q, this->qnl);
}

torch::Tensor Train::lossFunction(torch::Tensor enhancement, torch::Tensor target)
{
    return torch::sum(torch::pow(enhancement - target, 2))/this->nbatch;
}

void Train::train()
{
    std::cout << "train begin" << std::endl;
    auto dataset = OF_data(this->nn.inputs, this->enhancement).map(torch::data::transforms::Stack<>());
    auto data_loader = torch::data::make_data_loader(dataset, this->nbatch);

    // std::cout << "size" << dataset.size() << std::endl;

    torch::optim::SGD optimizer(this->nn.parameters(), 0.01);
    // std::cout << this->nn.parameters() << std::endl;
    for (size_t epoch = 1; epoch <= 100; ++epoch)
    {
        size_t batch_index = 0;
        for (auto& batch : *data_loader)
        {
            // std::cout << "inpt" <<batch.data << std::endl;
            // std::cout << "target" << batch.target << std::endl;
            optimizer.zero_grad();
            torch::Tensor prediction = this->nn.forward(batch.data);
            // std::cout << "prediction" << prediction << std::endl;
            // torch::Tensor loss = this->lossFunction(prediction, batch.target);
            torch::Tensor loss = torch::mse_loss(prediction, batch.target);
            // std::cout << "loss" << loss << std::endl;
            // std::cout << "batch_index" << batch_index << std::endl;
            loss.backward();
            // break;
            optimizer.step();
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(this->nn.parameters(), "net.pt");
            }
        }
        // break;
    }
}

int main()
{
    Train train;
    train.setPara(64000, 6, 10);
    train.loadData();
    train.initNN();
    train.train();
}