// Include the header for this source file
#include "genetic_algorithm.h"
#include "coupledstrip_lib.h"

#include <random>
#include <numeric>
#include <iostream>
#include <stdexcept>
#include <format>

void print_progress_bar(int total, int current, double value) {
    constexpr int bar_width = 20; // Width of the progress bar
    float progress = static_cast<float>(current) / total;

    std::cout << "[";
    int pos = static_cast<int>(bar_width * progress);
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << static_cast<int>(progress * 100.0f) << "(" << value <<")" << "%\r"; // \r returns cursor to the beginning of the line
    std::cout.flush(); // Ensure the output is printed immediately
}

namespace GA{
    /*
    *******************************************************
    *                      Constructor                    *
    *******************************************************
    */
    GeneticAlgorithm::GeneticAlgorithm(CSA::CoupledstripArrangement& arrangement,
        Eigen::ArrayXd g_left_start,
        Eigen::ArrayXd x_left_start,
        Eigen::ArrayXd g_right_start,
        Eigen::ArrayXd x_right_start, 
        int population_size, 
        int num_generations) 
        : 
        arrangement(arrangement),
        g_left_start(g_left_start),
        x_left_start(x_left_start), 
        g_right_start(g_right_start),
        x_right_start(x_right_start), 
        population_size(population_size), 
        num_generations(num_generations){
            #pragma omp parallel
            {
                #pragma omp single
                {
                    rng_engines.resize(omp_get_num_threads());
                }

                int tid = omp_get_thread_num();
                std::seed_seq seed{std::random_device{}(), static_cast<unsigned int>(tid)};
                rng_engines[tid] = std::mt19937(seed);
            }
            parent_index_dist = std::uniform_int_distribution<>(0, population_size - 1);
        }
    
    /*
    *******************************************************
    *               Delt->Curve->Delta                    *
    *******************************************************
    */
     Eigen::ArrayXd GeneticAlgorithm::curve_to_delta(const Eigen::ArrayXd& curve, size_t vector_size, bool decreasing){
        
        Eigen::ArrayXd delta = curve.bottomRows(vector_size - 1) - curve.topRows(vector_size - 1);
        
        if (decreasing){
            // -dx to maintain positive values
            delta = -1*delta;
        }

        return delta;
     }

     Eigen::ArrayXd GeneticAlgorithm::delta_to_curve(const Eigen::Ref<const Eigen::VectorXd>& delta, size_t vector_size, bool decreasing){
        Eigen::ArrayXd curve(vector_size);
        Eigen::ArrayXd cumsum(delta.size());

        if (decreasing){
            curve(0) = 1;
            std::partial_sum(delta.data(), delta.data() + delta.size(), cumsum.data());
            // deltas are positive so subtract
            curve.bottomRows(vector_size-1) = curve(0) - cumsum;

            return curve;
        }
        // Increasing case
        curve(0) = 0;
        std::partial_sum(delta.data(), delta.data() + delta.size(), cumsum.data());
        curve.bottomRows(vector_size-1) = curve(0) + cumsum;

        return curve;
     }

    /*
    *******************************************************
    *               Initial Population                    *
    *******************************************************
    */
    void GeneticAlgorithm::initialize_population(Eigen::MatrixXd& population_left, Eigen::MatrixXd& population_right, double& noise_scale){
        
        // Vector size (with the expectation that left and right side have same size)
        size_t delta_size = g_left_start.size() - 1; // Population is made up of deltas

        // Random uniform distribution between -1 to 1 scaled to 0 to 1 and further scaled to create random noise
        Eigen::MatrixXd random_noise_left = (
            (noise_scale * 
                0.5 * (Eigen::MatrixXd::Ones(delta_size, population_size) + Eigen::MatrixXd::Random(delta_size, population_size))
            ).array() * population_left.array()
        ).matrix();

        Eigen::MatrixXd random_noise_right = (
            (noise_scale * 
                0.5 * (Eigen::MatrixXd::Ones(delta_size, population_size) + Eigen::MatrixXd::Random(delta_size, population_size))
            ).array() * population_right.array()
        ).matrix();

        // Add noise to create the initial population
        population_left.noalias() += random_noise_left;
        population_right.noalias() += random_noise_right;

        // Limit the initial population within the boundary(0 to 1, as the curves are always scaled to be in this range)
        population_left = population_left.array().min(1).max(0).matrix();
        population_right = population_right.array().min(1).max(0).matrix();

    }

    /*
    *******************************************************
    *                 Fitness Operator                    *
    *******************************************************
    */
    double GeneticAlgorithm::calculate_fitness(Eigen::ArrayXd& individual_left,Eigen::ArrayXd& individual_right){
        // Vector size (with the expectation that left and right side have same size)
        size_t vector_size = individual_left.size();

        // Since the entire curve is given for the crossover make sure the boundary values are correct
        individual_left(0) = 0.0;
        individual_left(vector_size-1) = 1.0;
        individual_right(0) = 1.0;
        individual_right(vector_size-1) = 0.0;
        
        // Check for necessary condition
        if (!CSA::is_monotone(individual_left,CSA::MonotoneType::Increasing) || !CSA::is_monotone(individual_right,CSA::MonotoneType::Decreasing)){
            return CSA::degree_monotone(individual_left, individual_right); // high energy value since necessary condition failed
        }

        // Energy calculation
        Eigen::ArrayXd vn = CSA::calculate_potential_coeffs(arrangement.V0,
            arrangement.width_micrstr,
            arrangement.space_bw_strps,
            arrangement.hw_arra,
            arrangement.num_fs,
            individual_left,
            x_left_start,
            individual_right,
            x_right_start);

        return CSA::calculate_energy(arrangement.er1,
            arrangement.er2,
            arrangement.hw_arra,
            arrangement.ht_arra,
            arrangement.ht_subs,
            arrangement.num_fs,
            vn);
    }
    
    /*
    *******************************************************
    *                 Parent Selection                    *
    *******************************************************
    */
    size_t GeneticAlgorithm::select_parent(const Eigen::ArrayXd& fitness_array, const int& thread_id) {
        // Tournament selection with size 4
        size_t best_candidate_idx;
        size_t competetor_idx;

        // Do tournament selection
        std::mt19937& rng = rng_engines[thread_id];
        best_candidate_idx = parent_index_dist(rng);

        for (int i : {1,2,3,4}){
            competetor_idx = parent_index_dist(rng);

            if (fitness_array[competetor_idx] < fitness_array[best_candidate_idx]) {
                std::swap(best_candidate_idx, competetor_idx);
            }
        }

        return best_candidate_idx;
    }

    /*
    *******************************************************
    *                    SBX Crossover                    *
    *******************************************************
    */
    void GeneticAlgorithm::crossover(Eigen::VectorXd& parent1, Eigen::VectorXd& parent2, Eigen::Ref<Eigen::VectorXd> child1, Eigen::Ref<Eigen::VectorXd> child2, double eta){
        
        size_t parent_size = parent1.size();
        double exponent = 1.0 / (eta + 1.0);

        // Generate a vector of random numbers
        Eigen::ArrayXd u = 0.5 * (Eigen::ArrayXd::Random(parent_size) + 1); // Random numbers between 0 and 1

        // The following arithmetic has 1.0 - u in the denominator and hence it is important that u never has 1.0 as value
        double epsilon = std::numeric_limits<double>::epsilon();
        u = u.cwiseMax(epsilon).cwiseMin(1.0 - epsilon); // Ensure u is not 1.0 or 0.0

        Eigen::ArrayXd beta(u.size());
        Eigen::Array<bool, Eigen::Dynamic, 1> mask = (u <= 0.5); // mask array
        // Compute both cases for the mask
        Eigen::ArrayXd beta_case_ulessthan  = (2.0 * u).pow(exponent);
        Eigen::ArrayXd beta_case_umorethan = (1.0 / (2.0 * (1.0 - u))).pow(exponent);
        beta = mask.select(beta_case_ulessthan, beta_case_umorethan);

        // Calculate children
        child1 = 0.5 * ((1.0 + beta) * parent1.array() + (1.0 - beta) * parent2.array()).matrix();
        child2 = 0.5 * ((1.0 - beta) * parent1.array() + (1.0 + beta) * parent2.array()).matrix();
    }

    /*
    *******************************************************
    *                     Reproduction                    *
    *******************************************************
    */
    Eigen::VectorXi GeneticAlgorithm::select_elites(Eigen::ArrayXd& fitness_array, size_t elite_size){
        Eigen::VectorXi sorted_idx = Eigen::VectorXi::LinSpaced(fitness_array.size(), 0, fitness_array.size()-1);

        // Minimisation problem
        std::partial_sort(sorted_idx.data(),
                      sorted_idx.data() + elite_size,
                      sorted_idx.data() + sorted_idx.size(),
                      [&](int a, int b) { return fitness_array(a) < fitness_array(b); });
        return sorted_idx.head(elite_size);
    }
    /*
    *******************************************************
    *                     Reproduction                    *
    *******************************************************
    */
    void GeneticAlgorithm::reproduce(Eigen::MatrixXd& population_left, Eigen::MatrixXd& population_right, Eigen::ArrayXd& fitness_array, double& noise_scale){
        // Vector size (with the expectation that left and right side have same size)
        size_t delta_size = g_left_start.size() - 1; // Population is made up of deltas

        // Create a random noise scaled matrix for mutation
        Eigen::MatrixXd new_population_left(delta_size, population_size);
        Eigen::MatrixXd new_population_right(delta_size, population_size);
        
        // Select Elites
        constexpr size_t elite_size = 10;
        Eigen::VectorXi elite_idx = select_elites(fitness_array, elite_size);

        // Retain Elites
        new_population_left.leftCols(elite_size)  = population_left(Eigen::all, elite_idx);
        new_population_right.leftCols(elite_size) = population_right(Eigen::all, elite_idx);

        #pragma omp parallel for
        for (int i=elite_size; i<population_size; i+=2){
            // Get the thread id for random number generation
            int thread_id = omp_get_thread_num();
            // Select the parents using tournament selection
            Eigen::VectorXd parent1_left = population_left.col(select_parent(fitness_array,thread_id));
            Eigen::VectorXd parent2_left = population_left.col(select_parent(fitness_array,thread_id));

            Eigen::VectorXd parent1_right = population_right.col(select_parent(fitness_array,thread_id));
            Eigen::VectorXd parent2_right = population_right.col(select_parent(fitness_array,thread_id));
            
            // Crossover with SBX crossover
            crossover(parent1_left,parent2_left,new_population_left.col(i),new_population_left.col(i+1)); 
            crossover(parent1_right,parent2_right,new_population_right.col(i),new_population_right.col(i+1));

        }

        // Limit the initial population within the boundary(0 to 1, as the curves are always scaled to be in this range)
        new_population_left = new_population_left.array().min(1).max(0).matrix();
        new_population_right = new_population_right.array().min(1).max(0).matrix();
        
        // Reset the values of population with new population
        population_left = new_population_left;
        population_right = new_population_right;
    }

    /*
    *******************************************************
    *              Main Optimisation Function             *
    *******************************************************
    */
    void GeneticAlgorithm::optimize(double& noise_scale, GAResult& result){
        size_t best_index = 0;
        double best_energy = result.energy_convergence(0);
        double previous_energy = result.best_energy;
        
        // Need the length for further processing
        size_t g_left_size = g_left_start.size();
        size_t g_right_size = g_right_start.size();
        
        // Engineer in the requirement for same number of coordinates
        if (g_left_size != g_right_size){
            throw std::invalid_argument(std::format("For efficient processing of GA optimisation,\
                the algorithm expects both left and right sides to have same number of coordinates.\
                g_left: {}, g_right: {}", 
                g_left_size, g_right_size));
        }

        // Vector size (with the expectation that left and right side have same size)
        size_t vector_size = g_left_size;

        // Get the delta arrays
        Eigen::ArrayXd delta_left = curve_to_delta(g_left_start,vector_size,false);
        Eigen::ArrayXd delta_right = curve_to_delta(g_right_start,vector_size,true);

        // Create an initial population matrix and fitness array
        Eigen::MatrixXd population_left = delta_left.replicate(1, population_size);
        Eigen::MatrixXd population_right = delta_right.replicate(1, population_size);
        
        // Add random noise and initialize the population for left and right sides
        initialize_population(population_left,population_right,noise_scale);
        
        // Array to hold fitness value corresponding to the individual in the population
        Eigen::ArrayXd fitness_array = Eigen::ArrayXd(population_size);

        // Iterate for num_generations steps
        for(size_t generation=1; generation<num_generations+1; ++generation){
            print_progress_bar(num_generations, generation, best_energy);

            // Fitness calculation
            #pragma omp parallel for
            for(int i=0; i<population_size; ++i){
                Eigen::ArrayXd individual_left = delta_to_curve(population_left.col(i),vector_size,false);
                Eigen::ArrayXd individual_right = delta_to_curve(population_right.col(i),vector_size,true);
                fitness_array[i] = calculate_fitness(individual_left,individual_right);
            }

            // Keep track
            best_energy = fitness_array.minCoeff(); // Get the best energy of the generation
            result.energy_convergence(generation) = best_energy; // Store the best energy of the generation
            
            // if (generation%200 == 0){
            //     // Check for convergence
            //     if (std::abs(previous_energy - best_energy) < previous_energy*1e-3){
            //         std::cout << "Converged at generation " << generation << "\n";
            //         break;
            //     }
            //     previous_energy = best_energy; // Update the previous energy
            // }
            // Reproduce
            reproduce(population_left, population_right, fitness_array, noise_scale);
        }

        // Final population fitness calculation
        #pragma omp parallel for
        for(int i=0; i<population_size; ++i){
            Eigen::ArrayXd individual_left = population_left.col(i).array();
            Eigen::ArrayXd individual_right = population_right.col(i).array();
            fitness_array[i] = calculate_fitness(individual_left,individual_right);
        }

        // Optimized curve and metrics
        best_energy = fitness_array.minCoeff(&best_index); // Get the best energy of the last generation
        
        result.best_curve_left = delta_to_curve(population_left.col(best_index), vector_size, false); // Store the best curve of the last generation
        result.best_curve_left(0) = 0.0;
        result.best_curve_left(vector_size-1) = 1.0;

        result.best_curve_right = delta_to_curve(population_right.col(best_index), vector_size, true);
        result.best_curve_right(0) = 1.0;
        result.best_curve_right(vector_size-1) = 0.0;

        result.best_energy = best_energy; // Store the best energy of the last generation
    }

} // end of namespace