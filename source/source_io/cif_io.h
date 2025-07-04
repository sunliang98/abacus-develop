#ifndef MODULE_IO_CIF_IO_H
#define MODULE_IO_CIF_IO_H
#include "source_cell/unitcell.h"
#include <string>
#include <vector>
#include <map>

namespace ModuleIO
{
/**
 * # CifParser
 * ## INTRODUCTION
 * ### In short
 * Tools for CIF file I/O.
 * 
 * ### Supported structure of CIF
 * A example (official template of CIF) is shown here (https://www.iucr.org/__data/iucr/ftp/pub/form.cif), 
 * but present impl. is ONLY capable to parse a CIF with ONE structure. If there are multiple structures
 * in the CIF file, unexpected behavior may occur.
 * 
 * A CIF file always contain two kinds of data structure, the first is simply the key-value pair, the second
 * is table which is lead by a keyword "loop_".
 * 
 * #### Key-value pair
 * key-value pair can have two types:
 * 
 * Type 1: the most general
 * _KEY1 VALUE1
 * _KEY2 VALUE2
 * ...
 * Type 2: text box
 * _KEY1
 * ;
 * very long text
 * VeRy LoNg TeXt
 * ...
 * ;
 * 
 * #### Table
 * The table in CIF must be lead by a keyword "loop_", and all titles of the columns will be list first, 
 * then all the rest are the values of the table. For example:
 * loop_
 * _KEY1 
 * _KEY2 
 * _KEY3
 * ...
 * VALUE11 VALUE21 VALUE31
 * VALUE12 VALUE22 VALUE32
 * ...
 * Once the number of values cannot be divided by the number of keys, will cause an assertion error.
 * 
 * 
 * ## Usage of this "class"
 * ### Read CIF file and store the information in a map
 * type the following line:
 * std::map<std::string, std::vector<std::string>> cif_info;
 * ModuleIO::CifParser::read("test.cif", cif_info);
 * 
 * Information of the cif file will stored in key-value pair manner, like
 * cif_info["_cell_length_a"] = {"2.46772428"};
 * cif_info["_cell_length_b"] = {"2.46772428"};
 * ...
 * cif_info["_atom_site_label"] = {"C1", "C2", ...};
 * cif_info["_atom_site_fract_x"] = {"0.00000000", "0.33333300", ...};
 * ...
 * NOTE1: only keys in table will have value with length larger than 1, otherwise all words will be
 * saved in the only element in std::vector<std::string>. For example please see unittest of this
 * class at source/source_io/test/cif_io_test.cpp.
 * NOTE2: One should note that for some cif files, the number will be in the format like
 * "0.00000000(1)", which means the uncertainty of the number is 1 in the last digit.
 * In this case, user should be careful to convert the string to double by its own.
 * 
 * ### Write CIF file with the given information.
 * ModuleIO::CifParser::write("test.cif", ...);
 * For detailed usage, see the static methods in the class. A to-be-deprecated usage is
 * simple as:
 * ModuleIO::CifParser cif("test.cif", ucell);
 * , where ucell is an instance of UnitCell. This usage is not encouraged.
 * 
 * ## Pythonization information
 * 1. function write
 * it is recommended to pythonize the 2nd overload of this function, the 3rd one will be
 * deprecated in the future.
 * 2. function read
 * this function can be directly pythonized.
 */
    class CifParser
    {
        public:
            CifParser() = delete; // I cannot see any necessity to have a default constructor
            CifParser(const std::string& fcif); // read the cif file and store the information
            ~CifParser() {} // actually do not need to do anything explicitly

            /**
             * @brief Print the CIF file from the given information.
             * 
             * @param fcif the output cif file name
             * @param abc_angles cell A B C angles in the form of [a, b, c, alpha, beta, gamma]
             * @param natom the number of atoms
             * @param atom_site_labels the atom site labels in the form of [atom1, atom2, ...]
             * @param atom_site_fract_coords the fractional coordinates of the atoms in the form of [x1, y1, z1, x2, y2, z2, ...]
             * @param title the title of the CIF file
             * @param data_tag the data tag of the CIF file
             * @param rank the rank which writes the CIF file
             * @param atom_site_occups the occupancies of the atoms in the form of [occup1, occup2, ...]
             * @param cell_formula_units_z the cell formula units Z
             */
            static void write(const std::string& fcif,
                              const double* abc_angles,
                              const int natom,
                              const std::string* atom_site_labels, // the one without numbers
                              const double* atom_site_fract_coords,
                              const std::string& title = "# generated by ABACUS",
                              const std::string& data_tag = "data_?",
                              const int rank = 0,
                              const double* atom_site_occups = nullptr, // may be this will be useful after impementation of VCA?
                              const std::string& cell_formula_units_z = "1");
            /**
             * @brief A Pybind wrapper for the static method write.
             * 
             * @param fcif the output cif file name
             * @param abc_angles cell A B C angles in the form of [a, b, c, alpha, beta, gamma]
             * @param atom_site_labels the atom site labels in the form of [atom1, atom2, ...]
             * @param atom_site_fract_coords the fractional coordinates of the atoms in the form of [x1, y1, z1, x2, y2, z2, ...]
             * @param title the title of the CIF file
             * @param data_tag the data tag of the CIF file
             * @param rank the rank which writes the CIF file
             * @param atom_site_occups the occupancies of the atoms in the form of [occup1, occup2, ...]
             * @param cell_formula_units_z the cell formula units Z
             */
            static void write(const std::string& fcif,
                              const std::vector<double>& abc_angles,
                              const std::vector<std::string>& atom_site_labels, // the one without numbers
                              const std::vector<double>& atom_site_fract_coords,
                              const std::string& title = "# generated by ABACUS",
                              const std::string& data_tag = "data_?",
                              const int rank = 0,
                              const std::vector<double>& atom_site_occups = {}, // may be this will be useful after impementation of VCA?
                              const std::string& cell_formula_units_z = "1");
            // the version with both spacegroup symmetry and point group symmetry ready
            // not for now, because it is too complicated. However it is a walk-around
            // way to fix issue #4998
            // static void write();

            /**
             * @brief Write CIF file with the whole UnitCell instance
             * 
             * @param fcif the output cif file name
             * @param ucell the input UnitCell instance
             * @param title the title of the CIF file
             * @param data_tag the data tag of the CIF file
             * @param rank the rank which writes the CIF file
             */
            static void write(const std::string& fcif,
                              const UnitCell& ucell,
                              const std::string& title = "# generated by ABACUS",
                              const std::string& data_tag = "data_?",
                              const int rank = 0);
            /**
             * @brief Read the CIF file and store the information in the map.
             * 
             * @param fcif the input cif file name
             * @param out the output map containing the information
             * @param rank the rank which reads the CIF file
             */
            static void read(const std::string& fcif,
                             std::map<std::string, std::vector<std::string>>& out,
                             const int rank = 0);

            /**
             * @brief get information by key from the stored information of cif file, if the key is not found, 
             * an empty vector will be returned.
             * 
             * @param key the key to search
             * @return std::vector<std::string>. Only columns in table will have more than one element, otherwise
             * all the information will be stored in the only element (the first).
             */
            std::vector<std::string> get(const std::string& key);

        private:
            // interface to ABACUS UnitCell impl.
            static void _unpack_ucell(const UnitCell& ucell,    // because ucell is too heavy...
                                      std::vector<double>& veca,
                                      std::vector<double>& vecb,
                                      std::vector<double>& vecc,
                                      int& natom,
                                      std::vector<std::string>& atom_site_labels,
                                      std::vector<double>& atom_site_fract_coords);

            // stores the information of the cif file
            std::map<std::string, std::vector<std::string>> raw_;
    };
} // namespace ModuleIO
#endif // MODULE_IO_CIF_IO_H