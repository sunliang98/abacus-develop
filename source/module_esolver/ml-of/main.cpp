#include "./train.h"
// #include <sstream>
// #include <math.h>
// #include "time.h"

int main()
{
    torch::set_default_dtype(caffe2::TypeMeta::fromScalarType(torch::kDouble));
    auto output = torch::get_default_dtype();
    std::cout << "Default type: " << output << std::endl;

    // train test
    Train train;
    train.readInput();
    train.loadData();
    if (train.check_pot)
    {
        train.potTest();
    }
    else
    {
        train.init();
        train.train();
    }
    // train.potTest();

    // torch::Tensor x = torch::ones({2,2});
    // x[0][0] = 0.;
    // x[1][0] = 2.;
    // x[1][1] = 3.;
    // x.requires_grad_(true);
    // std::cout << "x" << x << std::endl;
    // torch::Tensor y = x * x + x.t() * x.t();
    // std::cout << "y" << y << std::endl;
    // std::vector<torch::Tensor> tmp_pot;
    // std::vector<torch::Tensor> tmp_ipt;
    // std::vector<torch::Tensor> tmp_eye;
    // tmp_pot.push_back(y);
    // tmp_ipt.push_back(x);
    // tmp_eye.push_back(x);
    // // tmp_eye.push_back(torch::ones_like(y));
    // std::vector<torch::Tensor> tmp_grad;
    // tmp_grad = torch::autograd::grad(tmp_pot, tmp_ipt, tmp_eye, true, true, true);
    // std::cout << tmp_grad[0] << std::endl;

    // load test
    // std::shared_ptr<NN_OFImpl> nn = std::make_shared<NN_OFImpl>(64000, 6);
    // torch::load(nn, "./net.pt");
    // nn->setData(train.gamma, train.gammanl, train.p, train.pnl, train.q, train.qnl);
    // std::cout << nn->forward(nn->inputs);
}