#include <iostream>
#include <omp.h>
#include <vector>
#include <chrono>

// Custom Includes
#include "file_io.h"
#include "coupledstrip_lib.h"
#include "genetic_algorithm.h"
// External Includes
#include <Eigen/Dense>
// #include <pybind11/pybind11.h>
// #include <pybind11/eigen.h>

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
        std::cout<<"Starting Energy: " << energy_init <<" VAs\n";

        // GA Class Init
        std::cout<< "Started GA Opt" << std::endl;
        GA::GeneticAlgorithm GAProblem = GA::GeneticAlgorithm(arrangement,g_left,x_left,g_right,x_right,population_size,num_generations);
        GA::GAResult result = GA::GAResult(Eigen::ArrayXd::Zero(num_generations+1), 
        Eigen::VectorXd(g_left.size()), Eigen::VectorXd(g_left.size()), energy_init);
        result.energy_convergence(0) = energy_init;

        // Call the optimization
        double noise_scale=0.1;
        
        auto start = std::chrono::high_resolution_clock::now();
        GAProblem.optimize(noise_scale, result);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time: " << duration.count()/1000 << " s\n";
        std::cout << "Best energy: " << result.best_energy << " VAs\n";

        return result;
}

int main(){
    std::cout << "Genetic Algorithm Optimization" << std::endl;
    std::cout << "------------------------------" << std::endl;
    
    std::cout << "Using OpenMP with " << omp_get_max_threads() << " threads." << std::endl;
    // Set the number of threads for OpenMP
    omp_set_num_threads(omp_get_max_threads());

    /*
    *******************************************************
    *                      CASE D                         *
    *******************************************************
    */
    std::string filename = "../data/CaseD_predicted_curve.csv";
    std::cout<< "Reading g point values from: " << filename << std::endl;
    std::unordered_map<std::string, std::vector<double>> dataD = fileio::read_csv(filename);
    
    if (dataD["x_left"].empty() || dataD["g_left"].empty() ||
    dataD["x_right"].empty() || dataD["g_right"].empty()) {
    std::cerr << "Error: One or more input vectors are empty!" << std::endl;
    return 1;
    }

    // Coupled Strip arrangement TC#1 Case D
    double V0 = 1.0;
    double space_bw_strps = 200e-6;
    double width_micrstr = 150e-6;
    double ht_micrstr = 0;
    double hw_arra = 3e-3;
    double ht_arra = 2.76e-3;
    double ht_subs = 112e-6;
    double er1 = 1.0;
    double er2 = 4.5;
    int num_fs = 1000; 
    
    // Convert the x and g vectors to Eigen arrays
    Eigen::ArrayXd x_leftD = Eigen::Map<const Eigen::ArrayXd>(dataD["x_left"].data(), dataD["x_left"].size()); // Mx1
    Eigen::ArrayXd g_leftD = Eigen::Map<const Eigen::ArrayXd>(dataD["g_left"].data(), dataD["g_left"].size()); // Mx1
    Eigen::ArrayXd x_rightD = Eigen::Map<const Eigen::ArrayXd>(dataD["x_right"].data(), dataD["x_right"].size()); // Mx1
    Eigen::ArrayXd g_rightD = Eigen::Map<const Eigen::ArrayXd>(dataD["g_right"].data(), dataD["g_right"].size()); // Mx1

    int population_size = 100;
    int num_generations = 1000;
    GA::GAResult resultD = ga_optimize(V0,space_bw_strps,width_micrstr,ht_micrstr,hw_arra,ht_arra,ht_subs,er1,er2,num_fs,population_size,num_generations,
    x_leftD,g_leftD,x_rightD,g_rightD);
    
    std::cout<< "Finished GA starting store" << std::endl;
    std::vector<double> result_vec_leftD(resultD.best_curve_left.data(), resultD.best_curve_left.data() + resultD.best_curve_left.size());
    std::vector<double> result_vec_rightD(resultD.best_curve_right.data(), resultD.best_curve_right.data() + resultD.best_curve_right.size());
    std::unordered_map<std::string, std::vector<double>> result_dataD;
    result_dataD["x_left"] = dataD["x_left"];
    result_dataD["g_left"] = result_vec_leftD;
    result_dataD["x_right"] = dataD["x_right"];
    result_dataD["g_right"] = result_vec_rightD;
    std::string curve_output_filename = "../data/CaseD_optimized_curve.csv";
    fileio::write_csv(curve_output_filename, result_dataD);

    std::unordered_map<std::string, std::vector<double>> energy_history;
    energy_history["energy"] = std::vector<double>(resultD.energy_convergence.data(), resultD.energy_convergence.data() + resultD.energy_convergence.size());
    energy_history["generation"] = std::vector<double>(num_generations + 1);
    for (size_t i = 0; i < num_generations + 1; ++i) {
        energy_history["generation"][i] = static_cast<double>(i);
    }

    std::string history_output_filename = "../data/energy_history_CaseD.csv";
    fileio::write_csv(history_output_filename, energy_history);

    /*
    *******************************************************
    *                      CASE L                         *
    *******************************************************
    */
   // Read starting curve cased D
    std::string filename = "../data/CaseL_predicted_curve.csv";
    std::cout<< "Reading g point values from: " << filename << std::endl;
    std::unordered_map<std::string, std::vector<double>> dataL = fileio::read_csv(filename);
    
    if (dataL["x_left"].empty() || dataL["g_left"].empty() ||
    dataL["x_right"].empty() || dataL["g_right"].empty()) {
    std::cerr << "Error: One or more input vectors are empty!" << std::endl;
    return 1;
    }

    // Coupled Strip arrangement TC#1 Case D
    double V0 = 1.0;
    double space_bw_strps = 200e-6;
    double width_micrstr = 150e-6;
    double ht_micrstr = 0;
    double hw_arra = 3e-3;
    double ht_arra = 2.76e-3;
    double ht_subs = 112e-6;
    double er1 = 1.0;
    double er2 = 1.0;
    int num_fs = 1000; 
    
    // Convert the x and g vectors to Eigen arrays
    Eigen::ArrayXd x_leftL = Eigen::Map<const Eigen::ArrayXd>(dataL["x_left"].data(), dataL["x_left"].size()); // Mx1
    Eigen::ArrayXd g_leftL = Eigen::Map<const Eigen::ArrayXd>(dataL["g_left"].data(), dataL["g_left"].size()); // Mx1
    Eigen::ArrayXd x_rightL = Eigen::Map<const Eigen::ArrayXd>(dataL["x_right"].data(), dataL["x_right"].size()); // Mx1
    Eigen::ArrayXd g_rightL = Eigen::Map<const Eigen::ArrayXd>(dataL["g_right"].data(), dataL["g_right"].size()); // Mx1

    int population_size = 100;
    int num_generations = 1000;
    GA::GAResult resultL = ga_optimize(V0,space_bw_strps,width_micrstr,ht_micrstr,hw_arra,ht_arra,ht_subs,er1,er2,num_fs,population_size,num_generations,
    x_leftL,g_leftL,x_rightL,g_rightL);
    
    std::cout<< "Finished GA starting store" << std::endl;
    std::vector<double> result_vec_leftL(resultL.best_curve_left.data(), resultL.best_curve_left.data() + resultL.best_curve_left.size());
    std::vector<double> result_vec_rightL(resultL.best_curve_right.data(), resultL.best_curve_right.data() + resultL.best_curve_right.size());
    std::unordered_map<std::string, std::vector<double>> result_dataL;
    result_dataL["x_left"] = dataL["x_left"];
    result_dataL["g_left"] = result_vec_leftD;
    result_dataL["x_right"] = dataL["x_right"];
    result_dataL["g_right"] = result_vec_rightD;
    std::string curve_output_filename = "../data/CaseL_optimized_curve.csv";
    fileio::write_csv(curve_output_filename, result_dataL);

    std::unordered_map<std::string, std::vector<double>> energy_history;
    energy_history["energy"] = std::vector<double>(resultL.energy_convergence.data(), resultL.energy_convergence.data() + resultL.energy_convergence.size());
    energy_history["generation"] = std::vector<double>(num_generations + 1);
    for (size_t i = 0; i < num_generations + 1; ++i) {
        energy_history["generation"][i] = static_cast<double>(i);
    }

    std::string history_output_filename = "../data/energy_history_CaseL.csv";
    fileio::write_csv(history_output_filename, energy_history);

    /*
    *******************************************************
    *                      Metrics                        *
    *******************************************************
    */

    double cD = CSA::calculate_capacitance(V0, resultD.best_energy);
    double cL = CSA::calculate_capacitance(V0, resultL.best_energy);

    double zL = CSA::calculate_impedanceL(cL);
    double zD = CSA::calculate_impedanceD(cD,cL);

    double epsEff = CSA::calculate_epsilonEff(cD,cL);

    std::cout<<"\n----------------------------\n";
    std::cout<<"Capacitance D : " <<cD<< " F \n";
    std::cout<<"Capacitance L : " <<cL<< " F \n";
    std::cout<<"Impedance D   : " <<zD<< " Ohm \n";
    std::cout<<"Impedance L   : " <<zL<< " Ohm \n";
    std::cout<<"Epsilon Eff   : " <<epsEff<< "\n";
}