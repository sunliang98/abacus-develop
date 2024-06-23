//This file contains interfaces with libtorch,
//including loading of model and calculating gradients
//as well as subroutines that prints the results for checking

//The file contains 8 subroutines:
//1. cal_descriptor : obtains descriptors which are eigenvalues of pdm
//      by calling torch::linalg::eigh
//2. check_descriptor : prints descriptor for checking
//3. cal_gvx : gvx is used for training with force label, which is gradient of descriptors, 
//      calculated by d(des)/dX = d(pdm)/dX * d(des)/d(pdm) = gdmx * gvdm
//      using einsum
//4. check_gvx : prints gvx into gvx.dat for checking
//5. cal_gvdm : d(des)/d(pdm)
//      calculated using torch::autograd::grad
//6. load_model : loads model for applying V_delta
//7. cal_gedm : calculates d(E_delta)/d(pdm)
//      this is the term V(D) that enters the expression H_V_delta = |alpha>V(D)<alpha|
//      caculated using torch::autograd::grad
//8. check_gedm : prints gedm for checking
//9. cal_orbital_precalc : orbital_precalc is usted for training with orbital label, 
//                         which equals gvdm * orbital_pdm_shell, 
//                         orbital_pdm_shell[1,Inl,nm*nm] = dm_hl * overlap * overlap

#ifdef __DEEPKS

#include "LCAO_deepks.h"
#include "module_base/parallel_reduce.h"
#include "module_base/constants.h"
#include "module_hamilt_lcao/module_hcontainer/atom_pair.h"
#include "module_base/libm/libm.h"
#include "module_base/blas_connector.h"
#include "module_io/input.h"

void LCAO_Deepks::cal_descriptor_equiv(const int nat)
{
    ModuleBase::TITLE("LCAO_Deepks", "cal_descriptor_equiv");
    ModuleBase::timer::tick("LCAO_Deepks", "cal_descriptor_equiv");

    // a rather unnecessary way of writing this, but I'll do it for now
    if (!this->d_tensor.empty())
    {
        this->d_tensor.erase(this->d_tensor.begin(), this->d_tensor.end());
    }

    for(int iat = 0; iat < nat; iat++)
    {
        auto tmp = torch::zeros(des_per_atom,torch::kFloat64);
        std::memcpy(tmp.data_ptr(),pdm[iat],sizeof(double)*tmp.numel());
        this->d_tensor.push_back(tmp);
    }

    ModuleBase::timer::tick("LCAO_Deepks", "cal_descriptor_equiv");
}

//calculates descriptors from projected density matrices
void LCAO_Deepks::cal_descriptor(const int nat)
{
    ModuleBase::TITLE("LCAO_Deepks", "cal_descriptor");
    ModuleBase::timer::tick("LCAO_Deepks", "cal_descriptor");

    if(GlobalV::deepks_equiv)
    {
        this->cal_descriptor_equiv(nat);
        return;
    }

    //init pdm_tensor and d_tensor
    torch::Tensor tmp;

    //if pdm_tensor and d_tensor is not empty, clear it !!
    if (!this->d_tensor.empty())
    {
        this->d_tensor.erase(this->d_tensor.begin(), this->d_tensor.end());
    }
    if (!this->pdm_tensor.empty())
    {
        this->pdm_tensor.erase(this->pdm_tensor.begin(), this->pdm_tensor.end());
    }

    for (int inl = 0;inl < this->inlmax;++inl)
    {
        int nm = 2 * inl_l[inl] + 1;
        tmp = torch::ones({ nm, nm }, torch::TensorOptions().dtype(torch::kFloat64));
        for (int m1 = 0;m1 < nm;++m1)
        {
            for (int m2 = 0;m2 < nm;++m2)
            {
                tmp.index_put_({m1, m2}, this->pdm[inl][m1 * nm + m2]);
            }
        }
        //torch::Tensor tmp = torch::from_blob(this->pdm[inl], { nm, nm }, torch::requires_grad());
        tmp.requires_grad_(true);
        this->pdm_tensor.push_back(tmp);
        this->d_tensor.push_back(torch::ones({ nm }, torch::requires_grad(true)));
    }

    //cal d_tensor
    for (int inl = 0;inl < inlmax;++inl)
    {
        torch::Tensor vd;
        std::tuple<torch::Tensor, torch::Tensor> d_v(this->d_tensor[inl], vd);
        //d_v = torch::symeig(pdm_tensor[inl], /*eigenvalues=*/true, /*upper=*/true);
        d_v = torch::linalg::eigh(pdm_tensor[inl], /*uplo*/"U");
        d_tensor[inl] = std::get<0>(d_v);
    }
    ModuleBase::timer::tick("LCAO_Deepks", "cal_descriptor");
    return;
}

void LCAO_Deepks::check_descriptor(const UnitCell &ucell)
{
    ModuleBase::TITLE("LCAO_Deepks", "check_descriptor");
    if(GlobalV::MY_RANK!=0) return;
    std::ofstream ofs("descriptor.dat");
    ofs<<std::setprecision(10);
    if(!GlobalV::deepks_equiv)
    {
        for (int it = 0; it < ucell.ntype; it++)
        {
            for (int ia = 0; ia < ucell.atoms[it].na; ia++)
            {
                int iat=ucell.itia2iat(it,ia);
                ofs << ucell.atoms[it].label << " atom_index " << ia + 1 << " n_descriptor " << this->des_per_atom << std::endl;
                int id = 0;
                for(int inl=0;inl<inlmax/ucell.nat;inl++)
                {
                    int nm = 2*inl_l[inl]+1;
                    for(int im=0;im<nm;im++)
                    {
                        const int ind=iat*inlmax/ucell.nat+inl;
                        ofs << d_tensor[ind].index({im}).item().toDouble() << " ";
                        if (id % 8 == 7) ofs << std::endl;
                        id++;
                    }
                }   
                ofs << std::endl << std::endl;
            }
        }
    }
    else
    {
        for(int iat = 0; iat < ucell.nat; iat ++)
        {
            int it = ucell.iat2it[iat];
            ofs << ucell.atoms[it].label << " atom_index " << iat + 1 << " n_descriptor " << this->des_per_atom << std::endl;
            for(int i = 0; i < this->des_per_atom; i ++)
            {
                ofs << this->pdm[iat][i] << " ";
                if (i % 8 == 7) ofs << std::endl;
            }
            ofs << std::endl << std::endl;
        }
    }
    return;
}

//calculates gradient of descriptors from gradient of projected density matrices
void LCAO_Deepks::cal_gvx(const int nat)
{
    ModuleBase::TITLE("LCAO_Deepks","cal_gvx");
    //preconditions
    this->cal_gvdm(nat);
    if(!gdmr_vector.empty())
    {
        gdmr_vector.erase(gdmr_vector.begin(),gdmr_vector.end());
    }

    //gdmr_vector : nat(derivative) * 3 * inl(projector) * nm * nm
    if(GlobalV::MY_RANK==0)
    {
        //make gdmx as tensor
        int nlmax = this->inlmax/nat;
        for (int nl=0;nl<nlmax;++nl)
        {
            std::vector<torch::Tensor> bmmv;
            for (int ibt=0;ibt<nat;++ibt)
            {
                std::vector<torch::Tensor> xmmv;
                for (int i=0;i<3;++i)
                {
                    std::vector<torch::Tensor> ammv;
                    for (int iat=0; iat<nat; ++iat)
                    {
                        int inl = iat*nlmax + nl;
                        int nm = 2*this->inl_l[inl]+1;
                        std::vector<double> mmv;
                        for (int m1=0;m1<nm;++m1)
                        {
                            for(int m2=0;m2<nm;++m2)
                            {
                                if(i==0) mmv.push_back(this->gdmx[ibt][inl][m1*nm+m2]);
                                if(i==1) mmv.push_back(this->gdmy[ibt][inl][m1*nm+m2]);
                                if(i==2) mmv.push_back(this->gdmz[ibt][inl][m1*nm+m2]);
                            }
                        }//nm^2
                        torch::Tensor mm = torch::tensor(mmv, torch::TensorOptions().dtype(torch::kFloat64) ).reshape({nm, nm});    //nm*nm
                        ammv.push_back(mm);
                    }
                    torch::Tensor amm = torch::stack(ammv, 0);  //nat*nm*nm
                    xmmv.push_back(amm);
                }
                torch::Tensor bmm = torch::stack(xmmv, 0);  //3*nat*nm*nm
                bmmv.push_back(bmm); 
            }
            this->gdmr_vector.push_back(torch::stack(bmmv, 0)); //nbt*3*nat*nm*nm
        }
        assert(this->gdmr_vector.size()==nlmax);

        //einsum for each inl: 
        //gdmr_vector : b:nat(derivative) * x:3 * a:inl(projector) * m:nm * n:nm
        //gevdm_vector : a:inl * v:nm (descriptor) * m:nm (pdm, dim1) * n:nm (pdm, dim2)
        //gvx_vector : b:nat(derivative) * x:3 * a:inl(projector) * m:nm(descriptor)
        std::vector<torch::Tensor> gvx_vector;
        for (int nl = 0;nl<nlmax;++nl)
        {
            gvx_vector.push_back(at::einsum("bxamn, avmn->bxav", {this->gdmr_vector[nl], this->gevdm_vector[nl]}));
        }
        
        // cat nv-> \sum_nl(nv) = \sum_nl(nm_nl)=des_per_atom
        // concatenate index a(inl) and m(nm)
        this->gvx_tensor = torch::cat(gvx_vector, -1);

        assert(this->gvx_tensor.size(0) == nat);
        assert(this->gvx_tensor.size(1) == 3);
        assert(this->gvx_tensor.size(2) == nat);
        assert(this->gvx_tensor.size(3) == this->des_per_atom);
    }

    return;
}

void LCAO_Deepks::check_gvx(const int nat)
{
    std::stringstream ss;
    std::ofstream ofs_x;
    std::ofstream ofs_y;
    std::ofstream ofs_z;

    ofs_x<<std::setprecision(12);
    ofs_y<<std::setprecision(12);
    ofs_z<<std::setprecision(12);

    for(int ia=0;ia<nat;ia++)
    {
        ss.str("");
        ss<<"gvx_"<<ia<<".dat";
        ofs_x.open(ss.str().c_str());
        ss.str("");
        ss<<"gvy_"<<ia<<".dat";
        ofs_y.open(ss.str().c_str());
        ss.str("");
        ss<<"gvz_"<<ia<<".dat";
        ofs_z.open(ss.str().c_str());

        ofs_x << std::setprecision(10);
        ofs_y << std::setprecision(10);
        ofs_z << std::setprecision(10);
        
        for(int ib=0;ib<nat;ib++)
        {
            for(int inl=0;inl<inlmax/nat;inl++)
            {
                int nm = 2*inl_l[inl]+1;
                {
                    const int ind=ib*inlmax/nat+inl;
                    ofs_x << gvx_tensor.index({ia,0,ib,inl}).item().toDouble() << " ";
                    ofs_y << gvx_tensor.index({ia,1,ib,inl}).item().toDouble() << " ";
                    ofs_z << gvx_tensor.index({ia,2,ib,inl}).item().toDouble() << " ";
                }
            }
            ofs_x << std::endl;
            ofs_y << std::endl;
            ofs_z << std::endl;
        }
        ofs_x.close();
        ofs_y.close();
        ofs_z.close();        
    }
}

//calculates stress of descriptors from gradient of projected density matrices
void LCAO_Deepks::cal_gvepsl(const int nat)
{
    ModuleBase::TITLE("LCAO_Deepks","cal_gvepsl");
    //preconditions
    this->cal_gvdm(nat);
    if(!gdmepsl_vector.empty())
    {
        gdmepsl_vector.erase(gdmepsl_vector.begin(),gdmepsl_vector.end());
    }

    //gdmr_vector : nat(derivative) * 3 * inl(projector) * nm * nm
    if(GlobalV::MY_RANK==0)
    {
        //make gdmx as tensor
        int nlmax = this->inlmax/nat;
        for (int nl=0;nl<nlmax;++nl)
        {
            std::vector<torch::Tensor> bmmv;
            //for (int ipol=0;ipol<6;++ipol)
            //{
            //    std::vector<torch::Tensor> xmmv;
                for (int i=0;i<6;++i)
                {
                    std::vector<torch::Tensor> ammv;
                    for (int iat=0; iat<nat; ++iat)
                    {
                        int inl = iat*nlmax + nl;
                        int nm = 2*this->inl_l[inl]+1;
                        std::vector<double> mmv;
                        for (int m1=0;m1<nm;++m1)
                        {
                            for(int m2=0;m2<nm;++m2)
                            {
                                 mmv.push_back(this->gdm_epsl[i][inl][m1*nm+m2]);
                            }
                        }//nm^2
                        torch::Tensor mm = torch::tensor(mmv, torch::TensorOptions().dtype(torch::kFloat64) ).reshape({nm, nm});    //nm*nm
                        ammv.push_back(mm);
                    }
                    torch::Tensor bmm = torch::stack(ammv, 0);  //nat*nm*nm
                    bmmv.push_back(bmm);
                }
                //torch::Tensor bmm = torch::stack(xmmv, 0);  //3*nat*nm*nm
                //bmmv.push_back(bmm); 
            //}
            this->gdmepsl_vector.push_back(torch::stack(bmmv, 0)); //nbt*3*nat*nm*nm
        }
        assert(this->gdmepsl_vector.size()==nlmax);

        //einsum for each inl: 
        //gdmepsl_vector : b:npol * a:inl(projector) * m:nm * n:nm
        //gevdm_vector : a:inl * v:nm (descriptor) * m:nm (pdm, dim1) * n:nm (pdm, dim2)
        //gvepsl_vector : b:npol * a:inl(projector) * m:nm(descriptor)
        std::vector<torch::Tensor> gvepsl_vector;
        for (int nl = 0;nl<nlmax;++nl)
        {
            gvepsl_vector.push_back(at::einsum("bamn, avmn->bav", {this->gdmepsl_vector[nl], this->gevdm_vector[nl]}));
        }
        
        // cat nv-> \sum_nl(nv) = \sum_nl(nm_nl)=des_per_atom
        // concatenate index a(inl) and m(nm)
        this->gvepsl_tensor = torch::cat(gvepsl_vector, -1);
        assert(this->gvepsl_tensor.size(0) == 6);
        assert(this->gvepsl_tensor.size(1) == nat);
        assert(this->gvepsl_tensor.size(2) == this->des_per_atom);
    }

    return;
}

//dDescriptor / dprojected density matrix
void LCAO_Deepks::cal_gvdm(const int nat)
{
    ModuleBase::TITLE("LCAO_Deepks", "cal_gvdm");
    if(!gevdm_vector.empty())
    {
        gevdm_vector.erase(gevdm_vector.begin(),gevdm_vector.end());
    }
    //cal gevdm(d(EigenValue(D))/dD)
    int nlmax = inlmax/nat;
    for (int nl=0;nl<nlmax;++nl)
    {
        std::vector<torch::Tensor> avmmv;
        for (int iat = 0;iat<nat;++iat)
        {
            int inl = iat*nlmax+nl;
            int nm = 2*this->inl_l[inl]+1;
            //repeat each block for nm times in an additional dimension
            torch::Tensor tmp_x = this->pdm_tensor[inl].reshape({nm, nm}).unsqueeze(0).repeat({nm, 1, 1});
            //torch::Tensor tmp_y = std::get<0>(torch::symeig(tmp_x, true));
            torch::Tensor tmp_y = std::get<0>(torch::linalg::eigh(tmp_x, "U"));
            torch::Tensor tmp_yshell = torch::eye(nm, torch::TensorOptions().dtype(torch::kFloat64));
            std::vector<torch::Tensor> tmp_rpt;     //repeated-pdm-tensor (x)
            std::vector<torch::Tensor> tmp_rdt; //repeated-d-tensor (y)
            std::vector<torch::Tensor> tmp_gst; //gvx-shell
            tmp_rpt.push_back(tmp_x);
            tmp_rdt.push_back(tmp_y);
            tmp_gst.push_back(tmp_yshell);
            std::vector<torch::Tensor> tmp_res;
            tmp_res = torch::autograd::grad(tmp_rdt, tmp_rpt, tmp_gst, false, false, /*allow_unused*/true); //nm(v)**nm*nm
            avmmv.push_back(tmp_res[0]);
        }
        torch::Tensor avmm = torch::stack(avmmv, 0); //nat*nv**nm*nm
        this->gevdm_vector.push_back(avmm);
    }
    assert(this->gevdm_vector.size() == nlmax);
    return;
}

void LCAO_Deepks::load_model(const std::string& deepks_model)
{
    ModuleBase::TITLE("LCAO_Deepks", "load_model");

    try
	{
        this->module = torch::jit::load(deepks_model);
    }
    catch (const c10::Error& e)

	{
        std::cerr << "error loading the model" << std::endl;
        return;
    }
	return;
}

inline void generate_py_files(const int lmaxd, const int nmaxd)
{
    if(GlobalV::MY_RANK!=0) return;
    std::ofstream ofs("cal_gedm.py");
    ofs << "import torch" << std::endl;
    ofs << "import numpy as np" << std::endl << std::endl;
    ofs << "import sys" << std::endl;

    ofs << "from deepks.scf.enn.scf import BasisInfo" << std::endl;
    ofs << "from deepks.iterate.template_abacus import t_make_pdm" << std::endl;
    ofs << "from deepks.utils import load_yaml" << std::endl << std::endl;
    
    ofs << "basis = load_yaml('basis.yaml')['proj_basis']" << std::endl;
    ofs << "model = torch.jit.load(sys.argv[1])" << std::endl;
    ofs << "dm_eig = np.expand_dims(np.load('dm_eig.npy'),0)" << std::endl;
    ofs << "dm_eig = torch.tensor(dm_eig, dtype=torch.float64,requires_grad=True)" << std::endl << std::endl;
    
    ofs << "dm_flat,basis_info = t_make_pdm(dm_eig,basis)" << std::endl;
    ofs << "ec = model(dm_flat.double())" << std::endl;
    ofs << "gedm = torch.autograd.grad(ec,dm_eig,grad_outputs=torch.ones_like(ec))[0]" << std::endl << std::endl;

    ofs << "np.save('ec.npy',ec.double().detach().numpy())" << std::endl;
    ofs << "np.save('gedm.npy',gedm.double().numpy())" << std::endl;
    ofs.close();

    ofs.open("basis.yaml");
    ofs << "proj_basis:" << std::endl;
    for(int l = 0; l < lmaxd+1; l++)
    {
        ofs << "  - - " << l << std::endl;
        ofs << "    - [";
        for (int i = 0; i < nmaxd+1; i++)
        {
            ofs << "0";
            if (i != nmaxd)
            {
                ofs << ", ";
            }
        }
        ofs << "]" << std::endl;
    }

}

void LCAO_Deepks::cal_gedm_equiv(const int nat)
{
    ModuleBase::TITLE("LCAO_Deepks", "cal_gedm_equiv");

    this->save_npy_d(nat);
    generate_py_files(this->lmaxd, this->nmaxd);
    if(GlobalV::MY_RANK==0)
    {
        std::string cmd = "python cal_gedm.py " + INPUT.deepks_model;
        int stat = std::system(cmd.c_str());
        assert(stat == 0);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    this->load_npy_gedm(nat);
    std::string cmd = "rm -f cal_gedm.py basis.yaml";
    std::system(cmd.c_str());
}

//obtain from the machine learning model dE_delta/dDescriptor
void LCAO_Deepks::cal_gedm(const int nat)
{
    if(GlobalV::deepks_equiv)
    {
        this->cal_gedm_equiv(nat);
        return;
    }

    //using this->pdm_tensor
    ModuleBase::TITLE("LCAO_Deepks", "cal_gedm");

    //forward
    std::vector<torch::jit::IValue> inputs;
    
    //input_dim:(natom, des_per_atom)
    inputs.push_back(torch::cat(this->d_tensor, 0).reshape({ 1, nat, this->des_per_atom }));
    std::vector<torch::Tensor> ec;
    ec.push_back(module.forward(inputs).toTensor());    //Hartree
    this->E_delta = ec[0].item().toDouble() * 2;//Ry; *2 is for Hartree to Ry

    //cal gedm
    std::vector<torch::Tensor> gedm_shell;
    gedm_shell.push_back(torch::ones_like(ec[0]));
    this->gedm_tensor = torch::autograd::grad(ec, this->pdm_tensor, gedm_shell, /*retain_grad=*/true, /*create_graph=*/false, /*allow_unused=*/true);

    //gedm_tensor(Hartree) to gedm(Ry)
    for (int inl = 0;inl < inlmax;++inl)
    {
        int nm = 2 * inl_l[inl] + 1;
        for (int m1 = 0;m1 < nm;++m1)
        {
            for (int m2 = 0;m2 < nm;++m2)
            {
                int index = m1 * nm + m2;
                //*2 is for Hartree to Ry
                this->gedm[inl][index] = this->gedm_tensor[inl].index({ m1,m2 }).item().toDouble() * 2;
            }
        }
    }
    return;
}

void LCAO_Deepks::check_gedm()
{
    std::ofstream ofs("gedm.dat");
    for(int inl=0;inl<inlmax;inl++)
    {
        int nm = 2 * inl_l[inl] + 1;
        for (int m1 = 0;m1 < nm;++m1)
        {
            for (int m2 = 0;m2 < nm;++m2)
            {
                int index = m1 * nm + m2;
                //*2 is for Hartree to Ry
                ofs << this->gedm[inl][index] << " ";
            }
        }   
        ofs << std::endl;     
    }
}

// calculates orbital_precalc[1,NAt,NDscrpt] = gvdm * orbital_pdm_shell;
// orbital_pdm_shell[2,Inl,nm*nm] = dm_hl * overlap * overlap;
void LCAO_Deepks::cal_orbital_precalc(const std::vector<std::vector<ModuleBase::matrix>> &dm_hl,
    const int nat,
    const UnitCell &ucell,
    const LCAO_Orbitals &orb,
    Grid_Driver& GridD)
{
    ModuleBase::TITLE("LCAO_Deepks", "calc_orbital_precalc");
    
    this->cal_gvdm(nat);
    const double Rcut_Alpha = orb.Alpha[0].getRcut();
    this->init_orbital_pdm_shell(1);
   
    for (int T0 = 0; T0 < ucell.ntype; T0++)
    {
		Atom* atom0 = &ucell.atoms[T0]; 
        
        for (int I0 =0; I0< atom0->na; I0++)
        {
            const int iat = ucell.itia2iat(T0,I0);
            const ModuleBase::Vector3<double> tau0 = atom0->tau[I0];
            GridD.Find_atom(ucell, atom0->tau[I0] ,T0, I0);

            //trace alpha orbital
            std::vector<int> trace_alpha_row;
            std::vector<int> trace_alpha_col;
            int ib=0;
            for (int L0 = 0; L0 <= orb.Alpha[0].getLmax();++L0)
            {
                for (int N0 = 0;N0 < orb.Alpha[0].getNchi(L0);++N0)
                {
                    const int inl = this->inl_index[T0](I0, L0, N0);
                    const int nm = 2*L0+1;
            
                    for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                    {
                        for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                        {
                            trace_alpha_row.push_back(ib+m1);
                            trace_alpha_col.push_back(ib+m2);
                        }
                    }
                    ib+=nm;
                }
            }
            const int trace_alpha_size = trace_alpha_row.size();

            for (int ad1=0; ad1<GridD.getAdjacentNum()+1 ; ++ad1)
            {
                const int T1 = GridD.getType(ad1);
                const int I1 = GridD.getNatom(ad1);
                const int ibt1 = ucell.itia2iat(T1,I1);
                const ModuleBase::Vector3<double> tau1 = GridD.getAdjacentTau(ad1);
				const Atom* atom1 = &ucell.atoms[T1];
				const int nw1_tot = atom1->nw*GlobalV::NPOL;
				const double Rcut_AO1 = orb.Phi[T1].getRcut();

                const double dist1 = (tau1-tau0).norm() * ucell.lat0;
                if (dist1 >= Rcut_Alpha + Rcut_AO1)
                {
                    continue;
                }

                auto row_indexes = pv->get_indexes_row(ibt1);
                const int row_size = row_indexes.size();
                if(row_size == 0) continue;

                std::vector<double> s_1t(trace_alpha_size * row_size);
                std::vector<double> g_1dmt(trace_alpha_size * row_size, 0.0);
                for(int irow=0;irow<row_size;irow++)
                {
                    const double* row_ptr = this->nlm_save[iat][ad1][row_indexes[irow]][0].data();
                    for(int i=0;i<trace_alpha_size;i++)
                    {
                        s_1t[i * row_size + irow] = row_ptr[trace_alpha_row[i]];
                    }
                } 

				for (int ad2=0; ad2 < GridD.getAdjacentNum()+1 ; ad2++)
				{
					const int T2 = GridD.getType(ad2);
					const int I2 = GridD.getNatom(ad2);
					const int ibt2 = ucell.itia2iat(T2,I2);
					const ModuleBase::Vector3<double> tau2 = GridD.getAdjacentTau(ad2);
					const Atom* atom2 = &ucell.atoms[T2];
					const int nw2_tot = atom2->nw*GlobalV::NPOL;
					
					const double Rcut_AO2 = orb.Phi[T2].getRcut();
                	const double dist2 = (tau2-tau0).norm() * ucell.lat0;

					if (dist2 >= Rcut_Alpha + Rcut_AO2)
					{
						continue;
					}

                    auto col_indexes = pv->get_indexes_col(ibt2);
                    const int col_size = col_indexes.size();
                    if(col_size == 0) continue;

                    std::vector<double> s_2t(trace_alpha_size * col_size);
                    for(int icol=0;icol<col_size;icol++)
                    {
                        const double* col_ptr = this->nlm_save[iat][ad2][col_indexes[icol]][0].data();
                        for(int i=0;i<trace_alpha_size;i++)
                        {
                            s_2t[i * col_size + icol] = col_ptr[trace_alpha_col[i]];
                        }
                    }

                    std::vector<double> dm_array(row_size * col_size, 0.0);
                    for(int is=0;is<GlobalV::NSPIN;is++)
                    {
                        hamilt::AtomPair<double> dm_pair(ibt1, ibt2, 0, 0, 0, pv);
                        dm_pair.allocate(dm_array.data(), 0);
                        if(ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER())
                        {
                            dm_pair.add_from_matrix(dm_hl[0][is].c, pv->get_row_size(), 1.0, 1);
                        }
                        else
                        {
                            dm_pair.add_from_matrix(dm_hl[0][is].c, pv->get_col_size(), 1.0, 0);
                        }
                    }// is

                    //dgemm for s_2t and dm_array to get g_1dmt
                    constexpr char transa='T', transb='N';
                    const double gemm_alpha = 1.0, gemm_beta = 1.0;
                    dgemm_(
                        &transa, &transb, 
                        &row_size, 
                        &trace_alpha_size, 
                        &col_size, 
                        &gemm_alpha, 
                        dm_array.data(), 
                        &col_size,
                        s_2t.data(),  
                        &col_size,     
                        &gemm_beta,      
                        g_1dmt.data(),    
                        &row_size);
				}//ad2
                // do dot of g_1dmt and s_1t to get orbital_pdm_shell
                const double* p_g1dmt = g_1dmt.data();
                int ib=0, index=0, inc=1;
                for (int L0 = 0; L0 <= orb.Alpha[0].getLmax();++L0)
                {
                    for (int N0 = 0;N0 < orb.Alpha[0].getNchi(L0);++N0)
                    {
                        const int inl = this->inl_index[T0](I0, L0, N0);
                        const int nm = 2*L0+1;
                
                        for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                        {
                            for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                            {
                                orbital_pdm_shell[0][0][inl][m1*nm+m2] += 
                                    ddot_(&row_size, p_g1dmt+index*row_size, &inc, s_1t.data()+index*row_size, &inc);
                                index++;
                            }
                        }
                        ib+=nm;
                    }
                }
			}//ad1   
            
        }
    }
#ifdef __MPI
    for(int inl = 0; inl < this->inlmax; inl++)
    {
        Parallel_Reduce::reduce_all(this->orbital_pdm_shell[0][0][inl], (2 * this->lmaxd + 1) * (2 * this->lmaxd + 1));
    }
#endif    
    
    // transfer orbital_pdm_shell to orbital_pdm_shell_vector
    

    int nlmax = this->inlmax/nat;
   
    std::vector<torch::Tensor> orbital_pdm_shell_vector;
    for(int nl = 0; nl < nlmax; ++nl)
    {
        std::vector<torch::Tensor> kiammv;
        for(int iks = 0; iks < 1; ++iks)
        {
            std::vector<torch::Tensor> iammv;
            std::vector<torch::Tensor> ammv;
            for (int iat=0; iat<nat; ++iat)
            {
                int inl = iat*nlmax+nl;
                int nm = 2*this->inl_l[inl]+1;
                std::vector<double> mmv;
            
                for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                {
                    for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                    {
                        mmv.push_back(this->orbital_pdm_shell[iks][0][inl][m1*nm+m2]);
                    }
                }
                torch::Tensor mm = torch::tensor(mmv, torch::TensorOptions().dtype(torch::kFloat64) ).reshape({nm, nm});    //nm*nm
                ammv.push_back(mm);
            }
            torch::Tensor amm = torch::stack(ammv, 0); 
            iammv.push_back(amm);
            torch::Tensor iamm = torch::stack(iammv, 0);  //inl*nm*nm
            //orbital_pdm_shell_vector.push_back(iamm);
            kiammv.push_back(iamm);
        }
        torch::Tensor kiamm = torch::stack(kiammv, 0);  
        orbital_pdm_shell_vector.push_back(kiamm);
    }
       
    assert(orbital_pdm_shell_vector.size() == nlmax);
    
    //einsum for each nl: 
    std::vector<torch::Tensor> orbital_precalc_vector;
    for (int nl = 0; nl<nlmax; ++nl)
    {
        orbital_precalc_vector.push_back(at::einsum("kiamn, avmn->kiav", {orbital_pdm_shell_vector[nl], this->gevdm_vector[nl]}));
    }
       
    this->orbital_precalc_tensor = torch::cat(orbital_precalc_vector, -1);
    this->del_orbital_pdm_shell(1);
	return;
}

// calculates orbital_precalc[nks,2,NAt,NDscrpt] = gvdm * orbital_pdm_shell for multi-k case;
// orbital_pdm_shell[nks,2,Inl,nm*nm] = dm_hl_k * overlap * overlap;
void LCAO_Deepks::cal_orbital_precalc_k(const std::vector<std::vector<ModuleBase::ComplexMatrix>> &dm_hl_k,
    const int nat,
    const int nks,
    const std::vector<ModuleBase::Vector3<double>> &kvec_d,
    const UnitCell &ucell,
    const LCAO_Orbitals &orb,
    Grid_Driver& GridD)
{
    ModuleBase::TITLE("LCAO_Deepks", "calc_orbital_precalc_k");
    ModuleBase::timer::tick("LCAO_Deepks", "calc_orbital_precalc_k");
    
    this->cal_gvdm(nat);
    const double Rcut_Alpha = orb.Alpha[0].getRcut();
    this->init_orbital_pdm_shell(nks);
   
    for (int T0 = 0; T0 < ucell.ntype; T0++)
    {
		Atom* atom0 = &ucell.atoms[T0]; 
        
        for (int I0 =0; I0< atom0->na; I0++)
        {
            const int iat = ucell.itia2iat(T0,I0);
            const ModuleBase::Vector3<double> tau0 = atom0->tau[I0];
            GridD.Find_atom(ucell, atom0->tau[I0] ,T0, I0);

            //trace alpha orbital
            std::vector<int> trace_alpha_row;
            std::vector<int> trace_alpha_col;
            int ib=0;
            for (int L0 = 0; L0 <= orb.Alpha[0].getLmax();++L0)
            {
                for (int N0 = 0;N0 < orb.Alpha[0].getNchi(L0);++N0)
                {
                    const int inl = this->inl_index[T0](I0, L0, N0);
                    const int nm = 2*L0+1;
            
                    for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                    {
                        for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                        {
                            trace_alpha_row.push_back(ib+m1);
                            trace_alpha_col.push_back(ib+m2);
                        }
                    }
                    ib+=nm;
                }
            }
            const int trace_alpha_size = trace_alpha_row.size();

            for (int ad1=0; ad1<GridD.getAdjacentNum()+1 ; ++ad1)
            {
                const int T1 = GridD.getType(ad1);
                const int I1 = GridD.getNatom(ad1);
                const int ibt1 = ucell.itia2iat(T1,I1);
                const ModuleBase::Vector3<double> tau1 = GridD.getAdjacentTau(ad1);
                const double dist1 = (tau1-tau0).norm() * ucell.lat0;
                const double Rcut_AO1 = orb.Phi[T1].getRcut();
                if (dist1 >= Rcut_Alpha + Rcut_AO1)
                {
                    continue;
                }

				const Atom* atom1 = &ucell.atoms[T1];
				const int nw1_tot = atom1->nw*GlobalV::NPOL; 

                ModuleBase::Vector3<double> dR1(GridD.getBox(ad1).x, GridD.getBox(ad1).y, GridD.getBox(ad1).z);

                auto row_indexes = pv->get_indexes_row(ibt1);
                const int row_size = row_indexes.size();
                if(row_size == 0) continue;

                key_tuple key_1(ibt1,dR1.x,dR1.y,dR1.z);
                if(this->nlm_save_k[iat].find(key_1) == this->nlm_save_k[iat].end()) continue;
                std::vector<double> s_1t(trace_alpha_size * row_size);
                std::vector<double> g_1dmt(nks * trace_alpha_size * row_size, 0.0);
                for(int irow=0;irow<row_size;irow++)
                {
                    const double* row_ptr = this->nlm_save_k[iat][key_1][row_indexes[irow]][0].data();
                    for(int i=0;i<trace_alpha_size;i++)
                    {
                        s_1t[i * row_size + irow] = row_ptr[trace_alpha_row[i]];
                    }
                }

				for (int ad2=0; ad2 < GridD.getAdjacentNum()+1 ; ad2++)
				{
					const int T2 = GridD.getType(ad2);
					const int I2 = GridD.getNatom(ad2);
                    const int ibt2 = ucell.itia2iat(T2,I2);
                    if(ibt1>ibt2) continue;
					const ModuleBase::Vector3<double> tau2 = GridD.getAdjacentTau(ad2);
					const Atom* atom2 = &ucell.atoms[T2];
					const int nw2_tot = atom2->nw*GlobalV::NPOL;
                    ModuleBase::Vector3<double> dR2(GridD.getBox(ad2).x, GridD.getBox(ad2).y, GridD.getBox(ad2).z);
					
					const double Rcut_AO2 = orb.Phi[T2].getRcut();
                	const double dist2 = (tau2-tau0).norm() * ucell.lat0;

					if (dist2 >= Rcut_Alpha + Rcut_AO2)
					{
						continue;
					}

                    auto col_indexes = pv->get_indexes_col(ibt2);
                    const int col_size = col_indexes.size();
                    if(col_size == 0) continue;

                    std::vector<double> s_2t(trace_alpha_size * col_size);
                    key_tuple key_2(ibt2,dR2.x,dR2.y,dR2.z);
                    if(this->nlm_save_k[iat].find(key_2) == this->nlm_save_k[iat].end()) continue;
                    for(int icol=0;icol<col_size;icol++)
                    {
                        const double* col_ptr = this->nlm_save_k[iat][key_2][col_indexes[icol]][0].data();
                        for(int i=0;i<trace_alpha_size;i++)
                        {
                            s_2t[i * col_size + icol] = col_ptr[trace_alpha_col[i]];
                        }
                    }


                    std::vector<double> dm_array(row_size*nks * col_size, 0.0);
                    const int row_size_nks = row_size * nks;
                    for(int ik=0;ik<nks;ik++)
                    {
                        hamilt::AtomPair<double> dm_pair(ibt1, ibt2, (dR2-dR1).x, (dR2-dR1).y, (dR2-dR1).z, pv);
                        dm_pair.allocate(&dm_array[ik * row_size * col_size], 0);
                        const double arg = - (kvec_d[ik] * (dR2-dR1) ) * ModuleBase::TWO_PI;
                        double sinp, cosp;
                        ModuleBase::libm::sincos(arg, &sinp, &cosp);
                        const std::complex<double> kphase = std::complex<double>(cosp, sinp);
                        if(ModuleBase::GlobalFunc::IS_COLUMN_MAJOR_KS_SOLVER())
                        {
                            dm_pair.add_from_matrix(dm_hl_k[0][ik].c, pv->get_row_size(), kphase, 1);
                        }
                        else
                        {
                            dm_pair.add_from_matrix(dm_hl_k[0][ik].c, pv->get_col_size(), kphase, 0);
                        }
                    }// ik

                    //dgemm for s_2t and dm_array to get g_1dmt
                    constexpr char transa='T', transb='N';
                    double gemm_alpha = 1.0, gemm_beta = 1.0;
                    if(ibt1<ibt2) gemm_alpha = 2.0;
                    dgemm_(
                        &transa, &transb, 
                        &row_size_nks, 
                        &trace_alpha_size, 
                        &col_size, 
                        &gemm_alpha, 
                        dm_array.data(), 
                        &col_size,
                        s_2t.data(),  
                        &col_size,     
                        &gemm_beta,      
                        g_1dmt.data(),    
                        &row_size_nks);
				}//ad2
                for(int ik=0;ik<nks;ik++)
                {
                    // do dot of g_1dmt and s_1t to get orbital_pdm_shell
                    const double* p_g1dmt = g_1dmt.data() + ik * row_size;
                    int ib=0, index=0, inc=1;
                    for (int L0 = 0; L0 <= orb.Alpha[0].getLmax();++L0)
                    {
                        for (int N0 = 0;N0 < orb.Alpha[0].getNchi(L0);++N0)
                        {
                            const int inl = this->inl_index[T0](I0, L0, N0);
                            const int nm = 2*L0+1;
                    
                            for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                            {
                                for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                                {
                                    orbital_pdm_shell[ik][0][inl][m1*nm+m2] += 
                                        ddot_(&row_size, p_g1dmt+index*row_size*nks, &inc, s_1t.data()+index*row_size, &inc);
                                    index++;
                                }
                            }
                            ib+=nm;
                        }
                    }
                }
			}//ad1   
            
        }
    }

#ifdef __MPI
    for (int iks = 0; iks < nks; iks++)
    {
        for (int hl = 0; hl < 1; hl++)
        {
            for(int inl = 0; inl < this->inlmax; inl++)
            {
                Parallel_Reduce::reduce_all(this->orbital_pdm_shell[iks][hl][inl], (2 * this->lmaxd + 1)* (2 * this->lmaxd + 1));
            }
        }
    }
#endif    

    // transfer orbital_pdm_shell to orbital_pdm_shell_vector
    

    int nlmax = this->inlmax/nat;
   
    std::vector<torch::Tensor> orbital_pdm_shell_vector;
    
    for(int nl = 0; nl < nlmax; ++nl)
    {
        std::vector<torch::Tensor> kiammv;
        for(int iks = 0; iks < nks; ++iks)
        {
            std::vector<torch::Tensor> iammv;
            for(int hl=0; hl<1; ++hl)
            {
                std::vector<torch::Tensor> ammv;
                for (int iat=0; iat<nat; ++iat)
                {
                    int inl = iat*nlmax+nl;
                    int nm = 2*this->inl_l[inl]+1;
                    std::vector<double> mmv;
                
                    for (int m1=0; m1<nm; ++m1) // m1 = 1 for s, 3 for p, 5 for d
                    {
                        for (int m2=0; m2<nm; ++m2) // m1 = 1 for s, 3 for p, 5 for d
                        {
                            mmv.push_back(this->orbital_pdm_shell[iks][hl][inl][m1*nm+m2]);
                        }
                
                    }
                    torch::Tensor mm = torch::tensor(mmv, torch::TensorOptions().dtype(torch::kFloat64) ).reshape({nm, nm});    //nm*nm
                    ammv.push_back(mm);
                }
                torch::Tensor amm = torch::stack(ammv, 0); 
                iammv.push_back(amm);
            }
            torch::Tensor iamm = torch::stack(iammv, 0);  //inl*nm*nm
            kiammv.push_back(iamm);
        }
        torch::Tensor kiamm = torch::stack(kiammv, 0);
        orbital_pdm_shell_vector.push_back(kiamm);
    }
       
    
    assert(orbital_pdm_shell_vector.size() == nlmax);
        
    
    //einsum for each nl: 
    std::vector<torch::Tensor> orbital_precalc_vector;
    for (int nl = 0; nl<nlmax; ++nl)
    {
        orbital_precalc_vector.push_back(at::einsum("kiamn, avmn->kiav", {orbital_pdm_shell_vector[nl], this->gevdm_vector[nl]}));
    }
    this->orbital_precalc_tensor = torch::cat(orbital_precalc_vector, -1);
       
    this->del_orbital_pdm_shell(nks);
    ModuleBase::timer::tick("LCAO_Deepks", "calc_orbital_precalc_k");
	return;
}

#endif
