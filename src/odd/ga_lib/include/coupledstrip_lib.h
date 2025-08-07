#ifndef COUPLEDSTRIP_LIB_H
#define COUPLEDSTRIP_LIB_H

// Includes
#include <vector>

// External includes
#include <Eigen/Dense>

// Include the Struct and Functions within the namespace so that when invoked it can be used to identify
namespace CSA{
    struct CoupledstripArrangement {
        const double V0;
        const double space_bw_strps;
        const double width_micrstr;
        const double ht_micrstr;
        const double hw_arra;
        const double ht_arra;
        const double ht_subs;
        const double er1;
        const double er2;
        const int num_fs; 

        CoupledstripArrangement(
            const double V0, // Potential at the microstrip
            const double space_bw_strps, // Space between the two strips
            const double width_micrstr, // Width of microstrip in meters 
            const double ht_micrstr, // Height of microstrip in meters 
            const double hw_arra, // Half-width of array in meters 
            const double ht_arra, // Height of array in meters 
            const double ht_subs, // Height of substrate in meters 
            const double er1, // Relative permittivity of air
            const double er2, // Relative permittivity of substrate
            const int num_fs) // Number of Fourier coefficients
            : 
            V0(V0),
            space_bw_strps(space_bw_strps), 
            width_micrstr(width_micrstr), 
            ht_micrstr(ht_micrstr), 
            hw_arra(hw_arra), 
            ht_arra(ht_arra), 
            ht_subs(ht_subs), 
            er1(er1), 
            er2(er2), 
            num_fs(num_fs){}
    }; // Struct CSA

    /*
    *******************************************************
    *            Necessary Conditons Chheck               *
    *******************************************************
    */
    // Function to check whether the curve is monotone decreasing or increasing
    enum class MonotoneType {
        Decreasing,
        Increasing
    };
    
    bool is_monotone(const Eigen::ArrayXd& g, MonotoneType curve_type);

    // Function to check whether the curve is convex
    bool is_convex(const Eigen::ArrayXd& g);

    /*
    *******************************************************
    *            Potential & Potential Coeffs             *
    *******************************************************
    */
    Eigen::ArrayXd calculate_potential_coeffs(const double& V0,
        const double& width_micrstr,
        const double& space_bw_strps, 
        const double& hw_arra, 
        const int& num_fs, 
        const Eigen::ArrayXd& g_left, 
        const Eigen::ArrayXd& x_left,
        const Eigen::ArrayXd& g_right, 
        const Eigen::ArrayXd& x_right);
    
    Eigen::ArrayXd calculate_potential(const double& hw_arra,
        Eigen::ArrayXd& vn, 
        std::vector<double>& x);

    /*
    *******************************************************
    *                      Energy                         *
    *******************************************************
    */
   Eigen::ArrayXd logsinh(const Eigen::ArrayXd& vector);

    Eigen::ArrayXd logcosh(const Eigen::ArrayXd& vector);

    double calculate_energy(const double& er1,
        const double& er2,
        const double& hw_arra,
        const double& ht_arra,
        const double& ht_subs,
        const int& num_fs,
        Eigen::ArrayXd& vn);

} // Namespace CSA

#endif