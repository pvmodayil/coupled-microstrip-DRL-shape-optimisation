// Include the header for this source file
#include "coupledstrip_lib.h"
#include <iostream>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <format>

/*
*******************************************************
*                      Constants                      *
*******************************************************
*/
constexpr double PI = std::numbers::pi;
constexpr double E0 = 8.854187817e-12; // vacuum permittivity in F/m 
constexpr double C0 = 2.99792458e8; // Speed of light in vacuum in m/s
namespace CSA{
    /*
    *******************************************************
    *            Necessary Conditons Chheck               *
    *******************************************************
    */
    // Function to check whether the curve is monotone decreasing or increasing    
    bool is_monotone(const Eigen::ArrayXd& g, MonotoneType curve_type){
        size_t m = g.size(); // Size of the curve

        if (m < 2){
            throw std::invalid_argument("Not enough spline knots for further processing");
        }

        if (curve_type == MonotoneType::Decreasing){
            return (g.tail(m-1) <= g.head(m-1)).all();
        }
        else if (curve_type == MonotoneType::Increasing){
            return (g.tail(m-1) >= g.head(m-1)).all();
        }
        else {
            throw std::invalid_argument("Invalid monotonicity type");
        }

    }

    // Function to return degree of monotonicity
    double degree_monotone(const Eigen::ArrayXd& g_left, const Eigen::ArrayXd& g_right){
        size_t m = g_left.size();
        Eigen::ArrayXd dx_left = g_left.bottomRows(m-1) - g_left.topRows(m-1);

        m = g_right.size();
        Eigen::ArrayXd dx_right = g_right.bottomRows(m-1) - g_right.topRows(m-1);

        double degree =  2*m - 0.5*((dx_left > 0).count() + (dx_right < 0).count()); // Shouldnt be zero

        return degree;
    }

    // Function to check whether the curve is convex
    bool is_convex(const Eigen::ArrayXd& g){
        size_t m = g.size();

        // Check if the vector has at least 3 elemen
        if (m < 3) {
            throw std::invalid_argument("Not enough spline knots for further processing");
        }

        // Calculate the second differences
        Eigen::ArrayXd dx2(m - 2);
        // segment(start idx, length)
        dx2 = g.segment(2, m-2) 
            - 2.0 * g.segment(1, m-2) 
            + g.segment(0, m-2);

        // Check if all second differences are non-negative
        return (dx2 >= 0).all();
    }

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
        const Eigen::ArrayXd& x_right){
        // Validate arguments
        //============================
        if (g_left.size() != x_left.size()){
            throw std::invalid_argument(std::format("Dimensions of x-axis vector and g-point vector for left side do not match! g: {}, x: {}", 
                g_left.size(), x_left.size()));
        }

        if (g_right.size() != x_right.size()){
            throw std::invalid_argument(std::format("Dimensions of x-axis vector and g-point vector for right side do not match! g: {}, x: {}", 
                g_right.size(), x_right.size()));
        }

        // Repeating or Constant terms
        //============================
        double M = g_left.size();
        double N = g_right.size();
        double d = space_bw_strps/2;

        Eigen::ArrayXd n = (Eigen::ArrayXd::LinSpaced(num_fs, 1, num_fs)); // nx1

        Eigen::ArrayXd alpha = n*PI/hw_arra; // nx1

        Eigen::ArrayXd m = (g_left.bottomRows(M-1) - g_left.topRows(M-1)) /
                        (x_left.bottomRows(M-1) - x_left.topRows(M-1)); // M-1x1
        Eigen::ArrayXd m_prime = (g_right.bottomRows(N-1) - g_right.topRows(N-1)) /
                        (x_right.bottomRows(N-1) - x_right.topRows(N-1)); // N-1x1

        double outer_coeff = 2*V0/hw_arra;

        // vn1
        //=========================
        // This calculation results in 2D so the .array() should be maped to ArrayXXd datatype , here a final .matrix will do the job
        Eigen::MatrixXd sin_left = ((alpha.matrix() * x_left.bottomRows(M-1).matrix().transpose()).array().sin()
                                    - (alpha.matrix() * x_left.topRows(M-1).matrix().transpose()).array().sin()).matrix(); // nx1 x 1xM-1 = nxM-1
        
        Eigen::ArrayXd vn1 = (1/alpha.square())*(
            (sin_left * m.matrix()).array()
        ); // nx1 x (nxM-1 x M-1x1) = nx1

        // vn2
        //=========================
        Eigen::ArrayXd vn2 = (1/alpha)*(
            ( 
            ((alpha.matrix() * x_left.topRows(M-1).matrix().transpose()).array().cos()).matrix()
            * g_left.topRows(M-1).matrix()
            ).array()
            - ( 
            ((alpha.matrix() * x_left.bottomRows(M-1).matrix().transpose()).array().cos()).matrix()
            * g_left.bottomRows(M-1).matrix()
            ).array()
        ); // nx1 x (cos(nx1 x 1XM-1) x M-1x1) = nx1

        // vn3
        //=========================
        Eigen::ArrayXd vn3 = (1/alpha)*(
            (alpha*d).cos() - (alpha*(d+width_micrstr)).cos()
        ); // nx1

        // vn4
        //=========================
        Eigen::MatrixXd sin_right = ((alpha.matrix() * x_right.bottomRows(N-1).matrix().transpose()).array().sin()
                                    - (alpha.matrix() * x_right.topRows(N-1).matrix().transpose()).array().sin()).matrix(); // nx1 x 1xM-1 = nxM-1
        Eigen::ArrayXd vn4 = (1/alpha.square())*(
            (sin_right.matrix() * m_prime.matrix()).array()
        ); // nx1 x (nxN-1 x N-1x1) = nx1

        // vn5
        //=========================
        Eigen::ArrayXd vn5 = (1/alpha)*(
            ( 
            ((alpha.matrix() * x_right.topRows(N-1).matrix().transpose()).array().cos()).matrix()
            * g_right.topRows(N-1).matrix()
            ).array()
            - ( 
            ((alpha.matrix() * x_right.bottomRows(N-1).matrix().transpose()).array().cos()).matrix()
            * g_right.bottomRows(N-1).matrix()
            ).array()
        ); // nx1 x (cos(nx1 x 1XN-1) x M-1x1) = nx1

        Eigen::ArrayXd vn = outer_coeff*(vn1+vn2+vn3+vn4+vn5);

        return vn; // nx1
    }
    
    Eigen::ArrayXd calculate_potential(const double& hw_arra,
        Eigen::ArrayXd& vn, 
        std::vector<double>& x){
        
        // Input potential coefficients must not be empty
        if(vn.rows() == 0){
            throw std::invalid_argument("Potenntial coefficients vn is empty");
        } 

        size_t num_fs = vn.size();
        Eigen::ArrayXd n = (Eigen::ArrayXd::LinSpaced(num_fs, 1, num_fs)); // nx1

        Eigen::ArrayXd alpha = n*PI/hw_arra; // nx1

        Eigen::MatrixXd x_vals = Eigen::Map<const Eigen::MatrixXd>(x.data(), x.size(), 1); // Mx1

        Eigen::ArrayXd sin = (x_vals * alpha.matrix().transpose()).array().sin(); // Mx1 x 1xn =  Mxn

        Eigen::ArrayXd VF = (sin.matrix() * vn.matrix()).array(); // Mxn x nx1 = Mx1

        return VF;
    }

    /*
    *******************************************************
    *                      Energy                         *
    *******************************************************
    */
   // sinh = (e^x - e^(-x))/2 => ln(sinh(x)) = ln(e^x - e^(-x)) - ln(2)
   // cosh = (e^x + e^(-x))/2 => ln(cosh(x)) = ln(e^x + e^(-x)) - ln(2)
   Eigen::ArrayXd logsinh(const Eigen::ArrayXd& vector){
        Eigen::ArrayXd absolute_vector = vector.abs();

        // Take care of overflow with threshold
        Eigen::ArrayXd logsinh_result = (absolute_vector > 33.0).select(
            absolute_vector - std::log(2.0), 
            (vector.exp() - (-vector).exp()).log() - std::log(2.0)); // nx1
            
        return logsinh_result;
   }

    Eigen::ArrayXd logcosh(const Eigen::ArrayXd& vector){
        Eigen::ArrayXd absolute_vector = vector.abs();

        // Take care of overflow with threshold
        Eigen::ArrayXd logcosh_result = (absolute_vector > 33.0).select(
            absolute_vector - std::log(2.0), 
            (vector.exp() + (-vector).exp()).log() - std::log(2.0)); // nx1
            
        return logcosh_result;
    }

    double calculate_energy(const double& er1,
        const double& er2,
        const double& hw_arra,
        const double& ht_arra,
        const double& ht_subs,
        const int& num_fs,
        Eigen::ArrayXd& vn){
        
        // Repeating or Constant terms
        //============================
        double e1 = er1 * E0;
        double e2 = er2 * E0;
        Eigen::ArrayXd n = (Eigen::ArrayXd::LinSpaced(num_fs, 1, num_fs)); // nx1
        
        Eigen::ArrayXd alpha = n*PI/hw_arra; // nx1
        
        Eigen::ArrayXd coeff = (n*PI/4)*vn.square(); // nx1

        // W1
        //=========
        Eigen::ArrayXd theta1 = alpha*(ht_arra-ht_subs); // nx1
        Eigen::ArrayXd coth1 = (logcosh(theta1) - logsinh(theta1)).exp();
        double w1 = e1 * coeff.matrix().dot(coth1.matrix()); // nx1 . nx1 = 1

        // W2
        //=========
        Eigen::ArrayXd theta2 = alpha*ht_subs; // nx1
        Eigen::ArrayXd coth2 = (logcosh(theta2) - logsinh(theta2)).exp();
        double w2 = e2 * coeff.matrix().dot(coth2.matrix()); // nx1 . nx1 = 1

        double w12 = w1 + w2;

        return w12;
    }

    /*
    *******************************************************
    *      Capacitance, Impedance, Epsilon Effective      *
    *******************************************************
    */
    double calculate_capacitance(double V0, double W){
        return (2*W)/(V0*V0);
    }

    double calculate_impedanceL(double capacitanceL){
        return 376.62*E0/capacitanceL;
    }

    double calculate_impedanceD(double capacitanceD, double capacitanceL){
        return 376.62*E0/(std::sqrt(capacitanceL*capacitanceD));
    }

    double calculate_epsilonEff(double capacitanceD, double capacitanceL){
        return capacitanceD/capacitanceL;
    }
} // namespace CSA