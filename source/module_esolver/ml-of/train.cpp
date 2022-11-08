#include "./train.h"
#include <sstream>


Train::~Train()
{
    delete[] this->train_cell;
    delete[] this->train_a;
    delete[] this->validation_cell;
    delete[] this->validation_a;
}

void Train::readInput()
{
    std::ifstream ifs("nnINPUT", std::ios::in);
    if (!ifs)
    {
        std::cout << " Can't find the nnINPUT file." << std::endl;
        exit(0);
    }

    char word[80];
    int ierr = 0;

    ifs.rdstate();
    while (ifs.good())
    {
        ifs >> word;
        if (ifs.eof()) break;

        if (strcmp("fftdim", word) == 0)
        {
            this->read_value(ifs, this->fftdim);
        }
        else if (strcmp("nbatch", word) == 0)
        {
            this->read_value(ifs, this->nbatch);
        }
        else if (strcmp("ntrain", word) == 0)
        {
            this->read_value(ifs, this->ntrain);
            this->train_cell = new std::string[this->ntrain];
            this->train_a = new double[this->ntrain];
        }
        else if (strcmp("nvalidation", word) == 0)
        {
            this->read_value(ifs, this->nvalidation);
            if (this->nvalidation > 0)
            {
                this->validation_cell = new std::string[this->nvalidation];
                this->validation_a = new double[this->nvalidation];
            }
        }
        else if (strcmp("train_dir", word) == 0)
        {
            this->read_value(ifs, this->train_dir);
        }
        else if (strcmp("train_cell", word) == 0)
        {
            for (int i = 0; i < this->ntrain; ++i)
            {
                ifs >> this->train_cell[i];
            }
        }
        else if (strcmp("train_a", word) == 0)
        {
            for (int i = 0; i < this->ntrain; ++i)
            {
                ifs >> this->train_a[i];
            }
        }
        else if (strcmp("validation_dir", word) == 0)
        {
            this->read_value(ifs, this->validation_dir);
        }
        else if (strcmp("validation_cell", word) == 0 && this->nvalidation > 0)
        {
            for (int i = 0; i < this->ntrain; ++i)
            {
                ifs >> this->validation_cell[i];
            }
        }
        else if (strcmp("validation_a", word) == 0 && this->nvalidation > 0)
        {
            for (int i = 0; i < this->ntrain; ++i)
            {
                ifs >> this->validation_a[i];
            }
        }
        else if (strcmp("loss", word) == 0)
        {
            this->read_value(ifs, this->loss);
        }
        else if (strcmp("nepoch", word) == 0)
        {
            this->read_value(ifs, this->nepoch);
        }
        else if (strcmp("step_length", word) == 0)
        {
            this->read_value(ifs, this->step_length);
        }
        else if (strcmp("dump_fre", word) == 0)
        {
            this->read_value(ifs, this->dump_fre);
        }
        else if (strcmp("print_fre", word) == 0)
        {
            this->read_value(ifs, this->print_fre);
        }
        else if (strcmp("gamma", word) == 0)
        {
            this->read_value(ifs, this->ml_gamma);
        }
        else if (strcmp("p", word) == 0)
        {
            this->read_value(ifs, this->ml_p);
        }
        else if (strcmp("q", word) == 0)
        {
            this->read_value(ifs, this->ml_q);
        }
        else if (strcmp("gammanl", word) == 0)
        {
            this->read_value(ifs, this->ml_gammanl);
        }
        else if (strcmp("pnl", word) == 0)
        {
            this->read_value(ifs, this->ml_pnl);
        }
        else if (strcmp("qnl", word) == 0)
        {
            this->read_value(ifs, this->ml_qnl);
        }
    }

    std::cout << "Read nnINPUT done" << std::endl;

    this->nx = this->fftdim * this->fftdim * this->fftdim * this->ntrain;
    this->nx_vali = this->fftdim * this->fftdim * this->fftdim * this->nvalidation;
    this->nn_input_index = {{"gamma", -1}, {"p", -1}, {"q", -1}, {"gammanl", -1}, {"pnl", -1}, {"qnl", -1}};

    this->ninput = 0;

    this->rho = std::vector<double>(this->nx);
    if (this->nvalidation > 0) this->rho_vali = std::vector<double>(this->nx_vali);

    if (this->ml_gamma || this->ml_gammanl){
        this->gamma = std::vector<double>(this->nx);
        if (this->nvalidation > 0) this->gamma_vali = std::vector<double>(this->nx_vali);
        if (this->ml_gamma)
        {
            this->nn_input_index["gamma"] = this->ninput; 
            this->ninput++;
        } 
    }    
    if (this->ml_p || this->ml_pnl){
        this->p = std::vector<double>(this->nx);
        this->nablaRho = std::vector<std::vector<double> >(3, std::vector<double>(this->nx, 0.));
        if (this->nvalidation > 0) this->p_vali = std::vector<double>(this->nx_vali);
        if (this->nvalidation > 0) this->nablaRho_vali = std::vector<std::vector<double> >(3, std::vector<double>(this->nx_vali, 0.));
        if (this->ml_p)
        {
            this->nn_input_index["p"] = this->ninput;
            this->ninput++;
        }
    }
    if (this->ml_q || this->ml_qnl){
        this->q = std::vector<double>(this->nx);
        if (this->nvalidation > 0) this->q_vali = std::vector<double>(this->nx_vali);
        if (this->ml_q)
        {
            this->nn_input_index["q"] = this->ninput;
            this->ninput++;
        }
    }
    if (this->ml_gammanl){
        this->gammanl = std::vector<double>(this->nx);
        if (this->nvalidation > 0) this->gammanl_vali = std::vector<double>(this->nx_vali);
        this->nn_input_index["gammanl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_pnl){
        this->pnl = std::vector<double>(this->nx);
        if (this->nvalidation > 0) this->pnl_vali = std::vector<double>(this->nx_vali);
        this->nn_input_index["pnl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_qnl){
        this->qnl = std::vector<double>(this->nx);
        if (this->nvalidation > 0) this->qnl_vali = std::vector<double>(this->nx_vali);
        this->nn_input_index["qnl"] = this->ninput;
        this->ninput++;
    }
    std::cout << "ninput" << this->ninput << std::endl;
}

void Train::loadData()
{
    this->loadData(this->train_dir, this->nx, this->rho, this->gamma, this->p, this->q, this->gammanl, this->pnl, this->qnl, this->nablaRho, this->enhancement);
    if (this->nvalidation > 0)
    {
        this->loadData(this->validation_dir, this->nx_vali, this->rho_vali, this->gamma_vali, this->p_vali, this->q_vali,
                        this->gammanl_vali, this->pnl_vali, this->qnl_vali, this->nablaRho_vali, this->enhancement_vali);
        this->input_vali = torch::zeros({this->nx_vali, this->ninput});
        if (this->nn_input_index["gamma"] >= 0) this->input_vali.index({"...", this->nn_input_index["gamma"]}) = torch::tensor(gamma_vali);
        if (this->nn_input_index["p"] >= 0) this->input_vali.index({"...", this->nn_input_index["p"]}) = torch::tensor(p_vali);
        if (this->nn_input_index["q"] >= 0) this->input_vali.index({"...", this->nn_input_index["q"]}) = torch::tensor(q_vali);
        if (this->nn_input_index["gammanl"] >= 0) this->input_vali.index({"...", this->nn_input_index["gammanl"]}) = torch::tensor(gammanl_vali);
        if (this->nn_input_index["pnl"] >= 0) this->input_vali.index({"...", this->nn_input_index["pnl"]}) = torch::tensor(pnl_vali);
        if (this->nn_input_index["qnl"] >= 0) this->input_vali.index({"...", this->nn_input_index["qnl"]}) = torch::tensor(qnl_vali);
    }

    std::cout << "Load train set done" << std::endl;
}

void Train::loadData(
    std::string dir, 
    int nx,
    std::vector<double> &rho,
    std::vector<double> &gamma,
    std::vector<double> &p,
    std::vector<double> &q,
    std::vector<double> &gammanl,
    std::vector<double> &pnl,
    std::vector<double> &qnl,
    std::vector<std::vector<double> > &nablaRho,
    torch::Tensor &enhancement
)
{
    std::vector<long unsigned int> cshape = {(long unsigned) nx};
    bool fortran_order = false;
    npy::LoadArrayFromNumpy(dir+"/gamma.npy", cshape, fortran_order, rho);
    if (this->ml_gamma || this->ml_gammanl) npy::LoadArrayFromNumpy(dir + "/gamma.npy", cshape, fortran_order, gamma);
    if (this->ml_gammanl) npy::LoadArrayFromNumpy(dir + "/gammanl.npy", cshape, fortran_order, gammanl);
    if (this->ml_p || this->ml_pnl) npy::LoadArrayFromNumpy(dir + "/p.npy", cshape, fortran_order, p);
    if (this->ml_pnl) npy::LoadArrayFromNumpy(dir + "/pnl.npy", cshape, fortran_order, pnl);
    if (this->ml_q || this->ml_qnl) npy::LoadArrayFromNumpy(dir + "/q.npy", cshape, fortran_order, q);
    if (this->ml_qnl) npy::LoadArrayFromNumpy(dir + "/qnl.npy", cshape, fortran_order, qnl);

    std::vector<double> enhancement_;
    npy::LoadArrayFromNumpy(dir+"enhancement.npy", cshape, fortran_order, enhancement_);
    enhancement = torch::tensor(enhancement_);
    enhancement.resize_({nx, 1});
}

void Train::initNN()
{
    this->nn = std::make_shared<NN_OFImpl>(this->nx, this->ninput);
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