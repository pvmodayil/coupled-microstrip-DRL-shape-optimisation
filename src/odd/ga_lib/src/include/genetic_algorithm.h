#ifndef GENETIC_ALGORITHM_H
#define GENETIC_ALGORITHM_H

// Includes
#include "coupledstrip_lib.h"
#include <random>
#include <omp.h>
// External includes
#include <Eigen/Dense>

namespace GA{
    // Result struct
    struct GAResult {
        Eigen::ArrayXd energy_convergence;
        Eigen::ArrayXd best_curve_left;
        Eigen::ArrayXd best_curve_right;
        double best_energy;

        GAResult(Eigen::ArrayXd energy_convergence, Eigen::VectorXd best_curve_left, Eigen::VectorXd best_curve_right, double best_energy)
            : energy_convergence(energy_convergence), best_curve_left(best_curve_left), best_curve_right(best_curve_right), best_energy(best_energy) {}
    };
    
    // Genetic Algorithm Class
    class GeneticAlgorithm {
        private:
            // GA Properties
            // ------------------------------------------------------
            Eigen::ArrayXd g_left_start;
            Eigen::ArrayXd x_left_start;
            Eigen::ArrayXd g_right_start;
            Eigen::ArrayXd x_right_start;
            int population_size;
            int num_generations;
            CSA::CoupledstripArrangement arrangement;

            // Random device
            std::vector<std::mt19937> rng_engines; // Vector of random number generators for parallel processing for thread safety
            //std::mt19937 rng;
            std::uniform_int_distribution<> parent_index_dist;
            
            // Functions
            // ------------------------------------------------------
            Eigen::ArrayXd curve_to_delta(const Eigen::ArrayXd& curve, size_t vector_size, bool decreasing);
            Eigen::ArrayXd delta_to_curve(const Eigen::Ref<const Eigen::VectorXd>& delta, size_t vector_size, bool decreasing);
            void initialize_population(Eigen::MatrixXd& population_left, Eigen::MatrixXd& population_right, double& noise_scale);

            double calculate_fitness(Eigen::ArrayXd& individual_left,Eigen::ArrayXd& individual_right);
            
            // Parent selection  
            size_t select_parent(const Eigen::ArrayXd& fitness_array, const int& thread_id);

            // Reproduction
            double get_eta(int generation, int num_generations);
            Eigen::VectorXi select_elites(Eigen::ArrayXd& fitness_array, size_t elite_size);
            void reproduce(Eigen::MatrixXd& population_left, Eigen::MatrixXd& population_right, Eigen::ArrayXd& fitness_array, double& eta);
            void crossover(Eigen::VectorXd& parent1, 
                Eigen::VectorXd& parent2, 
                Eigen::Ref<Eigen::VectorXd> child1, 
                Eigen::Ref<Eigen::VectorXd> child2, 
                double& eta);
            
        public:
            // Constructor
            GeneticAlgorithm(CSA::CoupledstripArrangement& arrangement, 
                Eigen::ArrayXd g_left_start,
                Eigen::ArrayXd x_left_start,
                Eigen::ArrayXd g_right_start,
                Eigen::ArrayXd x_right_start, 
                int population_size, 
                int num_generations);
            
            // Main function to run the optimization
            void optimize(double& noise_scale, GAResult& result);

    };

} // end of namespace

#endif