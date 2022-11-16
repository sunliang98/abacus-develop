#include "./train.h"


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
            this->train_dir = new std::string[this->ntrain];
            this->train_cell = new std::string[this->ntrain];
            this->train_a = new double[this->ntrain];
            this->train_volume = new double[this->ntrain];
        }
        else if (strcmp("nvalidation", word) == 0)
        {
            this->read_value(ifs, this->nvalidation);
            if (this->nvalidation > 0)
            {
                this->validation_dir = new std::string[this->nvalidation];
                this->validation_cell = new std::string[this->nvalidation];
                this->validation_a = new double[this->nvalidation];
                this->vali_volume = new double[this->nvalidation];
            }
        }
        else if (strcmp("train_dir", word) == 0)
        {
            for (int i = 0; i < this->ntrain; ++i)
            {
                ifs >> this->train_dir[i];
            }
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
            for (int i = 0; i < this->nvalidation; ++i)
            {
                ifs >> this->validation_dir[i];
            }
        }
        else if (strcmp("validation_cell", word) == 0 && this->nvalidation > 0)
        {
            for (int i = 0; i < this->nvalidation; ++i)
            {
                ifs >> this->validation_cell[i];
            }
        }
        else if (strcmp("validation_a", word) == 0 && this->nvalidation > 0)
        {
            for (int i = 0; i < this->nvalidation; ++i)
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

    this->nx = pow(this->fftdim, 3);
    this->nx_train = this->nx * this->ntrain;
    this->nx_vali = this->nx * this->nvalidation;
    this->nn_input_index = {{"gamma", -1}, {"p", -1}, {"q", -1}, {"gammanl", -1}, {"pnl", -1}, {"qnl", -1}};

    this->fft_grid_train = std::vector<std::vector<torch::Tensor>>(this->ntrain);
    this->fft_gg_train = std::vector<torch::Tensor>(this->ntrain);
    this->fft_kernel_train = std::vector<torch::Tensor>(this->ntrain);
    for (int i = 0; i < this->ntrain; ++i)
    {
        this->fft_grid_train[i] = std::vector<torch::Tensor>(3);
    }

    this->fft_grid_vali = std::vector<std::vector<torch::Tensor>>(this->nvalidation);
    this->fft_gg_vali = std::vector<torch::Tensor>(this->nvalidation);
    this->fft_kernel_vali = std::vector<torch::Tensor>(this->nvalidation);
    for (int i = 0; i < this->nvalidation; ++i)
    {
        this->fft_grid_vali[i] = std::vector<torch::Tensor>(3);
    }

    this->ninput = 0;

    this->rho = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    if (this->nvalidation > 0) this->rho_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
    this->enhancement = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    if (this->nvalidation > 0) this->enhancement_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
    this->pauli = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
    if (this->nvalidation > 0) this->pauli_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});

    if (this->ml_gamma || this->ml_gammanl){
        this->gamma = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
        if (this->nvalidation > 0) this->gamma_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        if (this->ml_gamma)
        {
            this->nn_input_index["gamma"] = this->ninput; 
            this->ninput++;
        } 
    }    
    if (this->ml_p || this->ml_pnl){
        this->p = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
        this->nablaRho = torch::zeros({this->ntrain, 3, this->fftdim, this->fftdim, this->fftdim});
        if (this->nvalidation > 0) this->p_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        if (this->nvalidation > 0) this->nablaRho_vali = torch::zeros({this->nvalidation, 3, this->fftdim, this->fftdim, this->fftdim});
        if (this->ml_p)
        {
            this->nn_input_index["p"] = this->ninput;
            this->ninput++;
        }
    }
    if (this->ml_q || this->ml_qnl){
        this->q = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
        if (this->nvalidation > 0) this->q_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        if (this->ml_q)
        {
            this->nn_input_index["q"] = this->ninput;
            this->ninput++;
        }
    }
    if (this->ml_gammanl){
        this->gammanl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
        if (this->nvalidation > 0) this->gammanl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->nn_input_index["gammanl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_pnl){
        this->pnl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
        if (this->nvalidation > 0) this->pnl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->nn_input_index["pnl"] = this->ninput;
        this->ninput++;
    }
    if (this->ml_qnl){
        this->qnl = torch::zeros({this->ntrain, this->fftdim, this->fftdim, this->fftdim});
        if (this->nvalidation > 0) this->qnl_vali = torch::zeros({this->nvalidation, this->fftdim, this->fftdim, this->fftdim});
        this->nn_input_index["qnl"] = this->ninput;
        this->ninput++;
    }

    std::cout << "ninput" << this->ninput << std::endl;
}
