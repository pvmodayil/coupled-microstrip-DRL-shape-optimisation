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
        Eigen::ArrayXd best_curve;
        double best_energy;

        GAResult(Eigen::ArrayXd energy_convergence, Eigen::VectorXd best_curve, double best_energy)
            : energy_convergence(energy_convergence), best_curve(best_curve), best_energy(best_energy) {}
    };
    
    // Genetic Algorithm Class
    class GeneticAlgorithm {
        private:
            // GA Properties
            // ------------------------------------------------------
            Eigen::ArrayXd starting_curveY;
            Eigen::ArrayXd starting_curveX;
            int population_size;
            int num_generations;
            double mutation_rate;
            CSA::CoupledstripArrangement arrangement;

            // Random device
            std::vector<std::mt19937> rng_engines; // Vector of random number generators for parallel processing for thread safety
            //std::mt19937 rng;
            std::uniform_int_distribution<> parent_index_dist;
            
            // Functions
            // ------------------------------------------------------
            Eigen::MatrixXd initialize_population(double& noise_scale);

            double calculate_fitness(Eigen::ArrayXd& individual);
            
            // Parent selection  
            size_t select_elites(const Eigen::ArrayXd& fitness_array);
            size_t select_parent(const Eigen::ArrayXd& fitness_array, const int& thread_id);

            // Reproduction
            Eigen::MatrixXd reproduce(Eigen::MatrixXd& population, Eigen::ArrayXd& fitness_array, double& noise_scale);
            void crossover(Eigen::VectorXd& parent1, 
                Eigen::VectorXd& parent2, 
                Eigen::Ref<Eigen::VectorXd> child1, 
                Eigen::Ref<Eigen::VectorXd> child2, 
                double eta=1.5);
            
        public:
            // Constructor
            GeneticAlgorithm(CSA::CoupledstripArrangement& arrangement, 
                Eigen::ArrayXd& starting_curveY,
                Eigen::ArrayXd& starting_curveX, 
                int population_size, 
                int num_generations, 
                double mutation_rate);
            
            // Main function to run the optimization
            void optimize(double& noise_scale, GAResult& result);

    };

} // end of namespace
#endif