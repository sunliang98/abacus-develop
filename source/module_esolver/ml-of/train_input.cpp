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
        else if (strcmp("feg_limit", word) == 0)
        {
            this->read_value(ifs, this->feg_limit);
        }
        else if (strcmp("coef_e", word) == 0)
        {
            this->read_value(ifs, this->coef_e);
        }
        else if (strcmp("coef_p", word) == 0)
        {
            this->read_value(ifs, this->coef_p);
        }
        else if (strcmp("coef_feg_e", word) == 0)
        {
            this->read_value(ifs, this->coef_feg_e);
        }
        else if (strcmp("coef_feg_p", word) == 0)
        {
            this->read_value(ifs, this->coef_feg_p);
        }
    }
    std::cout << "Read nnINPUT done" << std::endl;
}
