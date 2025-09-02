// External Includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <Eigen/Dense>

#include "coupledstrip_lib.h"
#include "genetic_algorithm.h"

namespace py = pybind11;

// GA Optimization frunction
GA::GAResult ga_optimize(const double V0,
            const double space_bw_strps,
            const double width_micrstr,
            const double ht_micrstr,
            const double hw_arra,
            const double ht_arra,
            const double ht_subs,
            const double er1,
            const double er2,
            const int num_fs,
            const int population_size,
            const int num_generations,
            Eigen::ArrayXd x_left,
            Eigen::ArrayXd g_left,
            Eigen::ArrayXd x_right,
            Eigen::ArrayXd g_right){
            
        // Arrangement struct init
        CSA::CoupledstripArrangement arrangement = CSA::CoupledstripArrangement(
            V0, space_bw_strps, width_micrstr, ht_micrstr, hw_arra, 
            ht_arra, ht_subs, er1, er2, num_fs); // N
        
        // Calculate init energy
        Eigen::ArrayXd vn = CSA::calculate_potential_coeffs(arrangement.V0,arrangement.width_micrstr,
        arrangement.space_bw_strps,arrangement.hw_arra,arrangement.num_fs,g_left,x_left,g_right,x_right);
        double energy_init = CSA::calculate_energy(arrangement.er1,arrangement.er2,arrangement.hw_arra,
        arrangement.ht_arra,arrangement.ht_subs,arrangement.num_fs,vn);

        // GA Class Init
        GA::GeneticAlgorithm GAProblem = GA::GeneticAlgorithm(arrangement,g_left,x_left,g_right,x_right,population_size,num_generations);
        GA::GAResult result = GA::GAResult(Eigen::ArrayXd::Zero(num_generations+1), 
        Eigen::VectorXd(g_left.size()), Eigen::VectorXd(g_left.size()), energy_init);
        result.energy_convergence(0) = energy_init;

        // Call the optimization
        double noise_scale=0.1;
        GAProblem.optimize(noise_scale, result);

        return result;
}

// To set maximum thread usage
void set_omp_to_max() {
    omp_set_num_threads(omp_get_max_threads());
}

py::dict ga_optimize_py(double V0,
                        double space_bw_strps,
                        double width_micrstr,
                        double ht_micrstr,
                        double hw_arra,
                        double ht_arra,
                        double ht_subs,
                        double er1,
                        double er2,
                        int num_fs,
                        int population_size,
                        int num_generations,
                        const Eigen::ArrayXd& x_left,
                        const Eigen::ArrayXd& g_left,
                        const Eigen::ArrayXd& x_right,
                        const Eigen::ArrayXd& g_right) 
{
    // call the actual C++ function
    GA::GAResult result = ga_optimize(V0, space_bw_strps, width_micrstr, ht_micrstr, hw_arra, ht_arra, ht_subs,
                                      er1, er2, num_fs, population_size, num_generations,
                                      x_left, g_left, x_right, g_right);

    py::dict py_result;
    py_result["energy_convergence"] = result.energy_convergence;
    py_result["best_curve_left"] = result.best_curve_left;
    py_result["best_curve_right"] = result.best_curve_right;
    py_result["best_energy"] = result.best_energy;
    return py_result;
}

PYBIND11_MODULE(ga_cpp, m) { // ga_cpp is the Python import name
    m.doc() = "Genetic Algorithm optimization for coupled microstrips"; 
    m.def("set_omp_to_max", &set_omp_to_max, "Set OMP to max threads");
    m.def("ga_optimize", &ga_optimize_py,
          py::arg("V0"),
          py::arg("space_bw_strps"),
          py::arg("width_micrstr"),
          py::arg("ht_micrstr"),
          py::arg("hw_arra"),
          py::arg("ht_arra"),
          py::arg("ht_subs"),
          py::arg("er1"),
          py::arg("er2"),
          py::arg("num_fs"),
          py::arg("population_size"),
          py::arg("num_generations"),
          py::arg("x_left"),
          py::arg("g_left"),
          py::arg("x_right"),
          py::arg("g_right"));
}