#include "kedf_ml.h"

void KEDF_ML::init_data(
    const int &nkernel,
    const bool &of_ml_gamma,
    const bool &of_ml_p,
    const bool &of_ml_q,
    const bool &of_ml_tanhp,
    const bool &of_ml_tanhq,
    const std::string &of_ml_gammanl_,
    const std::string &of_ml_pnl_,
    const std::string &of_ml_qnl_,
    const std::string &of_ml_xi_,
    const std::string &of_ml_tanhxi_,
    const std::string &of_ml_tanhxi_nl_,
    const std::string &of_ml_tanh_pnl_,
    const std::string &of_ml_tanh_qnl_,
    const std::string &of_ml_tanhp_nl_,
    const std::string &of_ml_tanhq_nl_
)
{
    bool* of_ml_gammanl = new bool[nkernel];
    bool* of_ml_pnl = new bool[nkernel];
    bool* of_ml_qnl = new bool[nkernel];
    bool* of_ml_xi = new bool[nkernel];
    bool* of_ml_tanhxi = new bool[nkernel];
    bool* of_ml_tanhxi_nl = new bool[nkernel];
    bool* of_ml_tanh_pnl = new bool[nkernel];
    bool* of_ml_tanh_qnl = new bool[nkernel];
    bool* of_ml_tanhp_nl = new bool[nkernel];
    bool* of_ml_tanhq_nl = new bool[nkernel];

    this->ml_data->split_string(of_ml_gammanl_, nkernel, false, of_ml_gammanl);
    this->ml_data->split_string(of_ml_pnl_, nkernel, false, of_ml_pnl);
    this->ml_data->split_string(of_ml_qnl_, nkernel, false, of_ml_qnl);
    this->ml_data->split_string(of_ml_xi_, nkernel, false, of_ml_xi);
    this->ml_data->split_string(of_ml_tanhxi_, nkernel, false, of_ml_tanhxi);
    this->ml_data->split_string(of_ml_tanhxi_nl_, nkernel, false, of_ml_tanhxi_nl);
    this->ml_data->split_string(of_ml_tanh_pnl_, nkernel, false, of_ml_tanh_pnl);
    this->ml_data->split_string(of_ml_tanh_qnl_, nkernel, false, of_ml_tanh_qnl);
    this->ml_data->split_string(of_ml_tanhp_nl_, nkernel, false, of_ml_tanhp_nl);
    this->ml_data->split_string(of_ml_tanhq_nl_, nkernel, false, of_ml_tanhq_nl);
    std::cout << "of_ml_gammanl  " << of_ml_gammanl[0] << "\t" << of_ml_gammanl[1] << std::endl;
    std::cout << "of_ml_xi  " << of_ml_xi[0] << "\t" << of_ml_xi[1] << std::endl;

    this->ninput = 0;

    // --------- semi-local descriptors ---------
    if (of_ml_gamma){
        this->descriptor_type.push_back("gamma");
        this->kernel_index.push_back(-1);
        ninput++;
    } 
    if (of_ml_p){
        this->descriptor_type.push_back("p");
        this->kernel_index.push_back(-1);
        ninput++;
    }
    if (of_ml_q){
        this->descriptor_type.push_back("q");
        this->kernel_index.push_back(-1);
        ninput++;
    }
    // --------- non-local descriptors ---------
    for (int ik = 0; ik < nkernel; ++ik)
    {
        if (of_ml_gammanl[ik]){
            this->descriptor_type.push_back("gammanl");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
        if (of_ml_pnl[ik]){
            this->descriptor_type.push_back("pnl");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
        if (of_ml_qnl[ik]){
            this->descriptor_type.push_back("qnl");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
        if (of_ml_xi[ik]){
            this->descriptor_type.push_back("xi");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
        if (of_ml_tanhxi[ik]){
            this->descriptor_type.push_back("tanhxi");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
        if (of_ml_tanhxi_nl[ik]){
            this->descriptor_type.push_back("tanhxi_nl");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
    }
    // --------- semi-local descriptors ---------
    if (of_ml_tanhp){
        this->descriptor_type.push_back("tanhp");
        this->kernel_index.push_back(-1);
        ninput++;
    }
    if (of_ml_tanhq){
        this->descriptor_type.push_back("tanhq");
        this->kernel_index.push_back(-1);
        ninput++;
    }
    // --------- non-local descriptors ---------
    for (int ik = 0; ik < nkernel; ++ik)
    {
        if (of_ml_tanh_pnl[ik]){
            this->descriptor_type.push_back("tanh_pnl");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
        if (of_ml_tanh_qnl[ik]){
            this->descriptor_type.push_back("tanh_qnl");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
        if (of_ml_tanhp_nl[ik]){
            this->descriptor_type.push_back("tanhp_nl");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
        if (of_ml_tanhq_nl[ik]){
            this->descriptor_type.push_back("tanhq_nl");
            this->kernel_index.push_back(ik);
            this->ninput++;
        }
    }

    this->descriptor2kernel = {{"gamma", {}},
                               {"p", {}},
                               {"q", {}},
                               {"tanhp", {}},
                               {"tanhq", {}},
                               {"gammanl", {}},
                               {"pnl", {}},
                               {"qnl", {}},
                               {"xi", {}},
                               {"tanhxi", {}},
                               {"tanhxi_nl", {}},
                               {"tanh_pnl", {}},
                               {"tanh_qnl", {}},
                               {"tanhp_nl", {}},
                               {"tanhq_nl", {}}};
    this->descriptor2index = this->descriptor2kernel;

    for (int i = 0; i < ninput; ++i)
    {
        this->descriptor2kernel[descriptor_type[i]].push_back(kernel_index[i]);
        // std::cout << "this->descriptor2kernel[descriptor_type[i]]    " << this->descriptor2kernel[descriptor_type[i]]
        //           << std::endl;
        this->descriptor2index[descriptor_type[i]].push_back(i);
        // std::cout << "this->descriptor2index[descriptor_type[i]]    " << this->descriptor2index[descriptor_type[i]]
        //           << std::endl;
    }
    std::cout << "descriptor2index    " << descriptor2index << std::endl;
    std::cout << "descriptor2kernel    " << descriptor2kernel << std::endl;

    this->ml_gamma = this->descriptor2index["gamma"].size() > 0;
    this->ml_p = this->descriptor2index["p"].size() > 0;
    this->ml_q = this->descriptor2index["q"].size() > 0;
    this->ml_tanhp = this->descriptor2index["tanhp"].size() > 0;
    this->ml_tanhq = this->descriptor2index["tanhq"].size() > 0;
    this->ml_gammanl = this->descriptor2index["gammanl"].size() > 0;
    this->ml_pnl = this->descriptor2index["pnl"].size() > 0;
    this->ml_qnl = this->descriptor2index["qnl"].size() > 0;
    this->ml_xi = this->descriptor2index["xi"].size() > 0;
    this->ml_tanhxi = this->descriptor2index["tanhxi"].size() > 0;
    this->ml_tanhxi_nl = this->descriptor2index["tanhxi_nl"].size() > 0;
    this->ml_tanh_pnl = this->descriptor2index["tanh_pnl"].size() > 0;
    this->ml_tanh_qnl = this->descriptor2index["tanh_qnl"].size() > 0;
    this->ml_tanhp_nl = this->descriptor2index["tanhp_nl"].size() > 0;
    this->ml_tanhq_nl = this->descriptor2index["tanhq_nl"].size() > 0;

    bool gene_gammanl_tot = false;
    bool gene_pnl_tot = false;
    bool gene_qnl_tot = false;
    // bool gene_xi_tot = false;
    // bool gene_tanhxi_tot = false;
    // bool gene_tanhxi_nl_tot = false;
    bool gene_tanh_pnl_tot = false;
    bool gene_tanh_qnl_tot = false;
    bool gene_tanhp_nl_tot = false;
    bool gene_tanhq_nl_tot = false;

    this->gene_data_label = {{"gamma", {}},
                               {"p", {}},
                               {"q", {}},
                               {"tanhp", {}},
                               {"tanhq", {}},
                               {"gammanl", {}},
                               {"pnl", {}},
                               {"qnl", {}},
                               {"xi", {}},
                               {"tanhxi", {}},
                               {"tanhxi_nl", {}},
                               {"tanh_pnl", {}},
                               {"tanh_qnl", {}},
                               {"tanhp_nl", {}},
                               {"tanhq_nl", {}}};

    for (std::string descriptor : {"gamma", "p", "q", "tanhp", "tanhq"})
    {
        this->gene_data_label[descriptor].push_back(0);
    }
    for (std::string descriptor : {"gammanl", "pnl", "qnl", "xi", "tanhxi", "tanhxi_nl",
                                    "tanh_pnl", "tanh_qnl", "tanhp_nl", "tanhq_nl"})
    {
        for (int ik = 0; ik < nkernel; ++ik)
        {
            this->gene_data_label[descriptor].push_back(0);
        }
    }

    for (int ik = 0; ik < nkernel; ++ik)
    {
        this->gene_data_label["pnl"][ik] = of_ml_pnl[ik] || of_ml_tanh_pnl[ik];
        this->gene_data_label["qnl"][ik] = of_ml_qnl[ik] || of_ml_tanh_qnl[ik];
        this->gene_data_label["tanhxi_nl"][ik] = of_ml_tanhxi_nl[ik];
        this->gene_data_label["tanhxi"][ik] = of_ml_tanhxi[ik] || of_ml_tanhxi_nl[ik];
        this->gene_data_label["xi"][ik] = of_ml_xi[ik] || this->gene_data_label["tanhxi"][ik];
        this->gene_data_label["gammanl"][ik] = of_ml_gammanl[ik] || this->gene_data_label["xi"][ik];
        this->gene_data_label["tanh_pnl"][ik] = of_ml_tanh_pnl[ik];
        this->gene_data_label["tanh_qnl"][ik] = of_ml_tanh_qnl[ik];
        this->gene_data_label["tanhp_nl"][ik] = of_ml_tanhp_nl[ik];
        this->gene_data_label["tanhq_nl"][ik] = of_ml_tanhq_nl[ik];
        // this->gene_data_label["pnl"][ik] = of_ml_pnl[ik] || of_ml_tanh_pnl[ik];

        gene_gammanl_tot = gene_gammanl_tot || this->gene_data_label["gammanl"][ik];
        gene_pnl_tot = gene_pnl_tot || this->gene_data_label["pnl"][ik];
        gene_qnl_tot = gene_qnl_tot || this->gene_data_label["qnl"][ik];
        // gene_xi_tot = gene_xi_tot || this->gene_data_label["xi"][ik];
        // gene_tanhxi_tot = gene_tanhxi_tot || this->gene_data_label["tanhxi"][ik];
        // gene_tanhxi_nl_tot = gene_tanhxi_nl_tot || this->gene_data_label["tanhxi_nl"][ik];
        gene_tanh_pnl_tot = gene_tanh_pnl_tot || this->gene_data_label["tanh_pnl"][ik];
        gene_tanh_qnl_tot = gene_tanh_qnl_tot || this->gene_data_label["tanh_qnl"][ik];
        gene_tanhp_nl_tot = gene_tanhp_nl_tot || this->gene_data_label["tanhp_nl"][ik];
        gene_tanhq_nl_tot = gene_tanhq_nl_tot || this->gene_data_label["tanhq_nl"][ik];
        
        // std::cout << "gene_gammanl    " << this->gene_data_label["gammanl"][ik] << std::endl;
        // std::cout << "gene_pnl    " << this->gene_data_label["pnl"][ik] << std::endl;
        // std::cout << "gene_qnl    " << this->gene_data_label["qnl"][ik] << std::endl;
        // std::cout << "gene_tanhxi_nl    " << this->gene_data_label["tanhxi_nl"][ik] << std::endl;
        // std::cout << "gene_tanhxi    " << this->gene_data_label["tanhxi"][ik] << std::endl;
        // std::cout << "gene_xi    " << this->gene_data_label["xi"][ik] << std::endl;
        // std::cout << "gene_tanh_pnl    " << this->gene_data_label["tanh_pnl"][ik] << std::endl;
        // std::cout << "gene_tanh_qnl    " << this->gene_data_label["tanh_qnl"][ik] << std::endl;
        // std::cout << "gene_tanhp_nl    " << this->gene_data_label["tanhp_nl"][ik] << std::endl;
        // std::cout << "gene_tanhq_nl    " << this->gene_data_label["tanhq_nl"][ik] << std::endl;
    }
    this->gene_data_label["gamma"][0] = of_ml_gamma || gene_gammanl_tot;
    this->gene_data_label["tanhp"][0] = of_ml_tanhp || gene_tanhp_nl_tot || gene_tanh_pnl_tot;
    this->gene_data_label["tanhq"][0] = of_ml_tanhq || gene_tanhq_nl_tot || gene_tanh_qnl_tot;
    this->gene_data_label["p"][0] = of_ml_p || this->gene_data_label["tanhp"][0] || gene_pnl_tot;
    this->gene_data_label["q"][0] = of_ml_q || this->gene_data_label["tanhq"][0] || gene_qnl_tot;


    if (this->gene_data_label["gamma"][0])   this->gamma = std::vector<double>(this->nx, 0.);
    if (this->gene_data_label["p"][0])
    {
        this->nablaRho = std::vector<std::vector<double> >(3, std::vector<double>(this->nx, 0.));
        this->p = std::vector<double>(this->nx, 0.);
    }
    if (this->gene_data_label["q"][0])       this->q = std::vector<double>(this->nx, 0.);
    if (this->gene_data_label["tanhp"][0])   this->tanhp = std::vector<double>(this->nx, 0.);
    if (this->gene_data_label["tanhq"][0])   this->tanhq = std::vector<double>(this->nx, 0.);

    for (int ik = 0; ik < nkernel; ++ik)
    {
        // int ktype = this->ml_data->kernel_type[ik];
        // double kscaling = input.kernel_scaling[ik];
        this->gammanl.push_back({});
        this->pnl.push_back({});
        this->qnl.push_back({});
        this->xi.push_back({});
        this->tanhxi.push_back({});
        this->tanhxi_nl.push_back({});
        this->tanh_pnl.push_back({});
        this->tanh_qnl.push_back({});
        this->tanhp_nl.push_back({});
        this->tanhq_nl.push_back({});

        if (this->gene_data_label["gammanl"][ik]){
            this->gammanl[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["pnl"][ik]){
            this->pnl[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["qnl"][ik]){
            this->qnl[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["xi"][ik]){
            this->xi[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["tanhxi"][ik]){
            this->tanhxi[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["tanhxi_nl"][ik]){
            this->tanhxi_nl[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["tanh_pnl"][ik]){
            this->tanh_pnl[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["tanh_qnl"][ik]){
            this->tanh_qnl[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["tanhp_nl"][ik]){
            this->tanhp_nl[ik] = std::vector<double>(this->nx, 0.);
        }
        if (this->gene_data_label["tanhq_nl"][ik]){
            this->tanhq_nl[ik] = std::vector<double>(this->nx, 0.);
        }
    }
}