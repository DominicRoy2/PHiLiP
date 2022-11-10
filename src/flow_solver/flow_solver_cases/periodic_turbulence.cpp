#include "periodic_turbulence.h"

#include <deal.II/base/function.h>
#include <stdlib.h>
#include <iostream>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/fe/fe_values.h>
#include "physics/physics_factory.h"
#include <deal.II/base/table_handler.h>
#include <deal.II/base/tensor.h>
#include "math.h"
#include <string>
#include <deal.II/base/quadrature_lib.h>

namespace PHiLiP {

namespace FlowSolver {

//=========================================================
// TURBULENCE IN PERIODIC CUBE DOMAIN
//=========================================================
template <int dim, int nstate>
PeriodicTurbulence<dim, nstate>::PeriodicTurbulence(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : PeriodicCubeFlow<dim, nstate>(parameters_input)
        , unsteady_data_table_filename_with_extension(this->all_param.flow_solver_param.unsteady_data_table_filename+".txt")
        , number_of_times_to_output_velocity_field(this->all_param.flow_solver_param.number_of_times_to_output_velocity_field)
        , output_velocity_field_at_fixed_times(this->all_param.flow_solver_param.output_velocity_field_at_fixed_times)
        , output_velocity_field_at_equidistant_nodes(this->all_param.flow_solver_param.output_velocity_field_at_equidistant_nodes)
        , output_vorticity_magnitude_field_in_addition_to_velocity(this->all_param.flow_solver_param.output_vorticity_magnitude_field_in_addition_to_velocity)
{
    // Get the flow case type
    using FlowCaseEnum = Parameters::FlowSolverParam::FlowCaseType;
    const FlowCaseEnum flow_type = this->all_param.flow_solver_param.flow_case_type;

    // Flow case identifiers
    this->is_taylor_green_vortex = (flow_type == FlowCaseEnum::taylor_green_vortex);
    this->is_decaying_homogeneous_isotropic_turbulence = (flow_type == FlowCaseEnum::decaying_homogeneous_isotropic_turbulence);
    this->is_viscous_flow = (this->all_param.pde_type != Parameters::AllParameters::PartialDifferentialEquation::euler);
    this->do_calculate_numerical_entropy= this->all_param.flow_solver_param.do_calculate_numerical_entropy;

    // Navier-Stokes object; create using dynamic_pointer_cast and the create_Physics factory
    PHiLiP::Parameters::AllParameters parameters_navier_stokes = this->all_param;
    parameters_navier_stokes.pde_type = Parameters::AllParameters::PartialDifferentialEquation::navier_stokes;
    this->navier_stokes_physics = std::dynamic_pointer_cast<Physics::NavierStokes<dim,dim+2,double>>(
                Physics::PhysicsFactory<dim,dim+2,double>::create_Physics(&parameters_navier_stokes));

    /* Initialize integrated quantities as NAN; 
       done as a precaution in the case compute_integrated_quantities() is not called
       before a member function of kind get_integrated_quantity() is called
     */
    std::fill(this->integrated_quantities.begin(), this->integrated_quantities.end(), NAN);

    /// For outputting velocity field
    this->index_of_current_desired_time_to_output_velocity_field = 0;
    if(output_velocity_field_at_fixed_times && (number_of_times_to_output_velocity_field > 0)) {
        this->output_velocity_field_times.reinit(number_of_times_to_output_velocity_field);
        
        // Get output_velocity_field_times from string
        const std::string output_velocity_field_times_string = this->all_param.flow_solver_param.output_velocity_field_times_string;
        std::string line = output_velocity_field_times_string;
        std::string::size_type sz1;
        output_velocity_field_times[0] = std::stod(line,&sz1);
        for(int i=1; i<number_of_times_to_output_velocity_field; ++i) {
            line = line.substr(sz1);
            sz1 = 0;
            output_velocity_field_times[i] = std::stod(line,&sz1);
        }

        // Get flow_field_quantity_filename_prefix
        flow_field_quantity_filename_prefix = "velocity";
        if(output_vorticity_magnitude_field_in_addition_to_velocity) {
            flow_field_quantity_filename_prefix += std::string("_vorticity");
        }
    }
}

template <int dim, int nstate>
void PeriodicTurbulence<dim,nstate>::display_additional_flow_case_specific_parameters() const
{
    this->pcout << "- - Courant-Friedrich-Lewy number: " << this->all_param.flow_solver_param.courant_friedrich_lewy_number << std::endl;
    std::string flow_type_string;
    if(this->is_taylor_green_vortex || this->is_decaying_homogeneous_isotropic_turbulence) {
        this->pcout << "- - Freestream Reynolds number: " << this->all_param.navier_stokes_param.reynolds_number_inf << std::endl;
        this->pcout << "- - Freestream Mach number: " << this->all_param.euler_param.mach_inf << std::endl;
    }
    this->display_grid_parameters();
}

template <int dim, int nstate>
double PeriodicTurbulence<dim,nstate>::get_constant_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    if(this->all_param.flow_solver_param.constant_time_step > 0.0) {
        const double constant_time_step = this->all_param.flow_solver_param.constant_time_step;
        this->pcout << "- - Using constant time step in FlowSolver parameters: " << constant_time_step << std::endl;
        return constant_time_step;
    } else {
        const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
        const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
        const double constant_time_step = this->all_param.flow_solver_param.courant_friedrich_lewy_number * approximate_grid_spacing;
        return constant_time_step;
    }
}

template <int dim, int nstate>
double PeriodicTurbulence<dim,nstate>::get_adaptive_time_step(std::shared_ptr<DGBase<dim,double>> dg) const
{
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
    const double approximate_grid_spacing = (this->domain_right-this->domain_left)/pow(number_of_degrees_of_freedom_per_state,(1.0/dim));
    const double cfl_number = this->all_param.flow_solver_param.courant_friedrich_lewy_number;
    const double time_step = cfl_number * approximate_grid_spacing / this->maximum_local_wave_speed;
    return time_step;
}

template <int dim, int nstate>
double PeriodicTurbulence<dim,nstate>::get_adaptive_time_step_initial(std::shared_ptr<DGBase<dim,double>> dg) const
{
    // initialize the maximum local wave speed
    update_maximum_local_wave_speed(*dg);
    // compute time step based on advection speed (i.e. maximum local wave speed)
    const double time_step = get_adaptive_time_step(dg);
    return time_step;
}

std::string get_padded_mpi_rank_string(const int mpi_rank_input) {
    // returns the mpi rank as a string with appropriate padding
    std::string mpi_rank_string = std::to_string(mpi_rank_input);
    const unsigned int length_of_mpi_rank_with_padding = 5;
    const int number_of_zeros = length_of_mpi_rank_with_padding - mpi_rank_string.length();
    mpi_rank_string.insert(0, number_of_zeros, '0');

    return mpi_rank_string;
}

template<int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::output_velocity_field(
    std::shared_ptr<DGBase<dim,double>> dg,
    const int output_file_index) const
{
    this->pcout << "     ... Writting velocity field ... " << std::flush;

    // NOTE: Same loop from read_values_from_file_and_project() in set_initial_condition.cpp
    
    // Get filename prefix based on output file index and the flow field quantity filename prefix
    const std::string filename_prefix = flow_field_quantity_filename_prefix + std::string("-") + std::to_string(output_file_index);

    // (1) Get filename based on MPI rank
    //-------------------------------------------------------------
    const int mpi_rank = dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    // -- Get padded mpi rank string
    const std::string mpi_rank_string = get_padded_mpi_rank_string(mpi_rank);
    // -- Assemble filename string
    const std::string filename_without_extension = filename_prefix + std::string("-") + mpi_rank_string;
    const std::string filename = filename_without_extension + std::string(".dat");
    //-------------------------------------------------------------

    // (2) Read file
    //-------------------------------------------------------------
    std::ofstream FILE (filename);
    
    // check that the file is open and write DOFs
    if (!FILE.is_open()) {
        this->pcout << "ERROR: Cannot open file " << filename << std::endl;
        std::abort();
    } else if(mpi_rank==0) {
        const unsigned int number_of_degrees_of_freedom_per_state = dg->dof_handler.n_dofs()/nstate;
        FILE << number_of_degrees_of_freedom_per_state << std::string("\n");
    }

    // write data
    std::array<double,nstate> soln_at_q;
    std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;

    const auto mapping = (*(dg->high_order_grid->mapping_fe_field));
    dealii::hp::MappingCollection<dim> mapping_collection(mapping);
    dealii::hp::FEValues<dim,dim> fe_values_collection(mapping_collection, dg->fe_collection, dg->volume_quadrature_collection, 
                                dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int max_dofs_per_cell = dg->dof_handler.get_fe_collection().max_dofs_per_cell();
    std::vector<dealii::types::global_dof_index> current_dofs_indices(max_dofs_per_cell);
    for (auto current_cell = dg->dof_handler.begin_active(); current_cell!=dg->dof_handler.end(); ++current_cell) {
        if (!current_cell->is_locally_owned()) continue;
    
        const int i_fele = current_cell->active_fe_index();
        const int i_quad = i_fele;
        const int i_mapp = 0;
        fe_values_collection.reinit (current_cell, i_quad, i_mapp, i_fele);
        const dealii::FEValues<dim,dim> &fe_values = fe_values_collection.get_present_fe_values();
        const unsigned int poly_degree = i_fele;
        const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
        const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
        
        current_dofs_indices.resize(n_dofs_cell);
        current_cell->get_dof_indices (current_dofs_indices);

        // for outputting equidistant nodes
        const dealii::FESystem<dim,dim> &fe_sys = dg->fe_collection[i_fele];
        dealii::Quadrature<dim> vol_quad_equidistant = dealii::QIterated<dim>(dealii::QTrapez<1>(),poly_degree);
        const std::vector<dealii::Point<dim>> &unit_equidistant_quad_pts = vol_quad_equidistant.get_points(); // all cells have same poly_degree

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            // write coordinates
            dealii::Point<dim> qpoint;
            if(output_velocity_field_at_equidistant_nodes) {
                // equidistant nodes
                qpoint = mapping.transform_unit_to_real_cell(current_cell,unit_equidistant_quad_pts[iquad]);
            } else {
                // GL nodes
                qpoint = (fe_values.quadrature_point(iquad));
            }
            for (int d=0; d<dim; ++d) {
                FILE << std::setprecision(17) << qpoint[d] << std::string(" ");
            }

            // get solution at qpoint
            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (int s=0; s<nstate; ++s) {
                for (int d=0; d<dim; ++d) {
                    soln_grad_at_q[s][d] = 0.0;
                }
            }
            
            for (unsigned int idof=0; idof<n_dofs_cell; ++idof) {
                const unsigned int istate = fe_values.get_fe().system_to_component_index(idof).first;
                
                if(output_velocity_field_at_equidistant_nodes) {
                    // at equidistant nodes
                    soln_at_q[istate] += dg->solution[current_dofs_indices[idof]] * fe_sys.shape_value_component(idof, unit_equidistant_quad_pts[iquad], istate);
                    soln_grad_at_q[istate] += dg->solution[current_dofs_indices[idof]] * fe_sys.shape_grad_component(idof, unit_equidistant_quad_pts[iquad], istate);
                } else {
                    // at GL nodes
                    soln_at_q[istate] += dg->solution[current_dofs_indices[idof]] * fe_values.shape_value_component(idof, iquad, istate);
                    soln_grad_at_q[istate] += dg->solution[current_dofs_indices[idof]] * fe_values.shape_grad_component(idof, iquad, istate);
                }
            }
            // write velocity field
            const dealii::Tensor<1,dim,double> velocity = this->navier_stokes_physics->compute_velocities(soln_at_q);
            for (int d=0; d<dim; ++d) {
                FILE << std::setprecision(17) << velocity[d] << std::string(" ");
            }
            // write vorticity magnitude field if desired
            if(output_vorticity_magnitude_field_in_addition_to_velocity) {
                const double vorticity_magnitude = this->navier_stokes_physics->compute_vorticity_magnitude(soln_at_q, soln_grad_at_q);
                FILE << std::setprecision(17) << vorticity_magnitude << std::string(" ");
            }
            FILE << std::string("\n"); // next line
        }
    }
    FILE.close();
    this->pcout << "done." << std::endl;
}

template<int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::update_maximum_local_wave_speed(DGBase<dim, double> &dg)
{    
    // Initialize the maximum local wave speed to zero
    this->maximum_local_wave_speed = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                                              dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
            }

            // Update the maximum local wave speed (i.e. convective eigenvalue)
            const double local_wave_speed = this->navier_stokes_physics->max_convective_eigenvalue(soln_at_q);
            if(local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
        }
    }
    this->maximum_local_wave_speed = dealii::Utilities::MPI::max(this->maximum_local_wave_speed, this->mpi_communicator);
}

template<int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::compute_and_update_integrated_quantities(DGBase<dim, double> &dg)
{
    std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integral_values;
    std::fill(integral_values.begin(), integral_values.end(), 0.0);
    
    // Initialize the maximum local wave speed to zero; only used for adaptive time step
    if(this->all_param.flow_solver_param.adaptive_time_step == true) this->maximum_local_wave_speed = 0.0;

    // Overintegrate the error to make sure there is not integration error in the error estimate
    int overintegrate = 10;
    dealii::QGauss<dim> quad_extra(dg.max_degree+1+overintegrate);
    dealii::FEValues<dim,dim> fe_values_extra(*(dg.high_order_grid->mapping_fe_field), dg.fe_collection[dg.max_degree], quad_extra,
                                              dealii::update_values | dealii::update_gradients | dealii::update_JxW_values | dealii::update_quadrature_points);

    const unsigned int n_quad_pts = fe_values_extra.n_quadrature_points;
    std::array<double,nstate> soln_at_q;
    std::array<dealii::Tensor<1,dim,double>,nstate> soln_grad_at_q;

    std::vector<dealii::types::global_dof_index> dofs_indices (fe_values_extra.dofs_per_cell);
    for (auto cell : dg.dof_handler.active_cell_iterators()) {
        if (!cell->is_locally_owned()) continue;
        fe_values_extra.reinit (cell);
        cell->get_dof_indices (dofs_indices);

        for (unsigned int iquad=0; iquad<n_quad_pts; ++iquad) {

            std::fill(soln_at_q.begin(), soln_at_q.end(), 0.0);
            for (int s=0; s<nstate; ++s) {
                for (int d=0; d<dim; ++d) {
                    soln_grad_at_q[s][d] = 0.0;
                }
            }
            for (unsigned int idof=0; idof<fe_values_extra.dofs_per_cell; ++idof) {
                const unsigned int istate = fe_values_extra.get_fe().system_to_component_index(idof).first;
                soln_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_value_component(idof, iquad, istate);
                soln_grad_at_q[istate] += dg.solution[dofs_indices[idof]] * fe_values_extra.shape_grad_component(idof,iquad,istate);
            }
            // const dealii::Point<dim> qpoint = (fe_values_extra.quadrature_point(iquad));

            std::array<double,NUMBER_OF_INTEGRATED_QUANTITIES> integrand_values;
            std::fill(integrand_values.begin(), integrand_values.end(), 0.0);
            integrand_values[IntegratedQuantitiesEnum::kinetic_energy] = this->navier_stokes_physics->compute_kinetic_energy_from_conservative_solution(soln_at_q);
            integrand_values[IntegratedQuantitiesEnum::enstrophy] = this->navier_stokes_physics->compute_enstrophy(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::pressure_dilatation] = this->navier_stokes_physics->compute_pressure_dilatation(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::deviatoric_strain_rate_tensor_magnitude_sqr] = this->navier_stokes_physics->compute_deviatoric_strain_rate_tensor_magnitude_sqr(soln_at_q,soln_grad_at_q);
            integrand_values[IntegratedQuantitiesEnum::strain_rate_tensor_magnitude_sqr] = this->navier_stokes_physics->compute_strain_rate_tensor_magnitude_sqr(soln_at_q,soln_grad_at_q);

            for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
                integral_values[i_quantity] += integrand_values[i_quantity] * fe_values_extra.JxW(iquad);
            }

            // Update the maximum local wave speed (i.e. convective eigenvalue) if using an adaptive time step
            if(this->all_param.flow_solver_param.adaptive_time_step == true) {
                const double local_wave_speed = this->navier_stokes_physics->max_convective_eigenvalue(soln_at_q);
                if(local_wave_speed > this->maximum_local_wave_speed) this->maximum_local_wave_speed = local_wave_speed;
            }
        }
    }
    this->maximum_local_wave_speed = dealii::Utilities::MPI::max(this->maximum_local_wave_speed, this->mpi_communicator);

    // update integrated quantities
    for(int i_quantity=0; i_quantity<NUMBER_OF_INTEGRATED_QUANTITIES; ++i_quantity) {
        this->integrated_quantities[i_quantity] = dealii::Utilities::MPI::sum(integral_values[i_quantity], this->mpi_communicator);
        this->integrated_quantities[i_quantity] /= this->domain_size; // divide by total domain volume
    }
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_integrated_kinetic_energy() const
{
    const double integrated_kinetic_energy = this->integrated_quantities[IntegratedQuantitiesEnum::kinetic_energy];
    // // Abort if energy is nan
    // if(std::isnan(integrated_kinetic_energy)) {
    //     this->pcout << " ERROR: Kinetic energy at time " << current_time << " is nan." << std::endl;
    //     this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
    //     std::abort();
    // }
    return integrated_kinetic_energy;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_integrated_enstrophy() const
{
    return this->integrated_quantities[IntegratedQuantitiesEnum::enstrophy];
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_vorticity_based_dissipation_rate() const
{
    const double integrated_enstrophy = this->integrated_quantities[IntegratedQuantitiesEnum::enstrophy];
    double vorticity_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        vorticity_based_dissipation_rate = this->navier_stokes_physics->compute_vorticity_based_dissipation_rate_from_integrated_enstrophy(integrated_enstrophy);
    }
    return vorticity_based_dissipation_rate;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_pressure_dilatation_based_dissipation_rate() const
{
    const double integrated_pressure_dilatation = this->integrated_quantities[IntegratedQuantitiesEnum::pressure_dilatation];
    return (-1.0*integrated_pressure_dilatation); // See reference (listed in header file), equation (57b)
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_deviatoric_strain_rate_tensor_based_dissipation_rate() const
{
    const double integrated_deviatoric_strain_rate_tensor_magnitude_sqr = this->integrated_quantities[IntegratedQuantitiesEnum::deviatoric_strain_rate_tensor_magnitude_sqr];
    double deviatoric_strain_rate_tensor_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        deviatoric_strain_rate_tensor_based_dissipation_rate = 
            this->navier_stokes_physics->compute_deviatoric_strain_rate_tensor_based_dissipation_rate_from_integrated_deviatoric_strain_rate_tensor_magnitude_sqr(integrated_deviatoric_strain_rate_tensor_magnitude_sqr);
    }
    return deviatoric_strain_rate_tensor_based_dissipation_rate;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_strain_rate_tensor_based_dissipation_rate() const
{
    const double integrated_strain_rate_tensor_magnitude_sqr = this->integrated_quantities[IntegratedQuantitiesEnum::strain_rate_tensor_magnitude_sqr];
    double strain_rate_tensor_based_dissipation_rate = 0.0;
    if (is_viscous_flow){
        strain_rate_tensor_based_dissipation_rate = 
            this->navier_stokes_physics->compute_strain_rate_tensor_based_dissipation_rate_from_integrated_strain_rate_tensor_magnitude_sqr(integrated_strain_rate_tensor_magnitude_sqr);
    }
    return strain_rate_tensor_based_dissipation_rate;
}

template<int dim, int nstate>
double PeriodicTurbulence<dim, nstate>::get_numerical_entropy(
        const std::shared_ptr <DGBase<dim, double>> dg
        ) const
{
    const double poly_degree = this->all_param.flow_solver_param.poly_degree;
    dealii::LinearAlgebra::distributed::Vector<double> mass_matrix_times_solution(dg->right_hand_side);
    if(this->all_param.use_inverse_mass_on_the_fly)
        dg->apply_global_mass_matrix(dg->solution,mass_matrix_times_solution);
    else
        dg->global_mass_matrix.vmult( mass_matrix_times_solution, dg->solution);

    const unsigned int n_dofs_cell = dg->fe_collection[poly_degree].dofs_per_cell;
    const unsigned int n_quad_pts = dg->volume_quadrature_collection[poly_degree].size();
    const unsigned int n_shape_fns = n_dofs_cell / nstate;
    //We have to project the vector of entropy variables because the mass matrix has an interpolation from solution nodes built into it.
    OPERATOR::vol_projection_operator<dim,2*dim> vol_projection(1, poly_degree, dg->max_grid_degree);
    vol_projection.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    OPERATOR::basis_functions<dim,2*dim> soln_basis(1, poly_degree, dg->max_grid_degree); 
    soln_basis.build_1D_volume_operator(dg->oneD_fe_collection_1state[poly_degree], dg->oneD_quadrature_collection[poly_degree]);

    dealii::LinearAlgebra::distributed::Vector<double> entropy_var_hat_global(dg->right_hand_side);
    std::vector<dealii::types::global_dof_index> dofs_indices (n_dofs_cell);


    for (auto cell = dg->dof_handler.begin_active(); cell!=dg->dof_handler.end(); ++cell) {
        if (!cell->is_locally_owned()) continue;
        cell->get_dof_indices (dofs_indices);

        std::array<std::vector<double>,nstate> soln_coeff;
        for(unsigned int idof=0; idof<n_dofs_cell; idof++){
            const unsigned int istate = dg->fe_collection[poly_degree].system_to_component_index(idof).first;
            const unsigned int ishape = dg->fe_collection[poly_degree].system_to_component_index(idof).second;
            if(ishape == 0)
                soln_coeff[istate].resize(n_shape_fns);
            soln_coeff[istate][ishape] = dg->solution(dofs_indices[idof]);
        }

        std::array<std::vector<double>,nstate> soln_at_q;
        for(int istate=0; istate<nstate; istate++){
            soln_at_q[istate].resize(n_quad_pts);
            soln_basis.matrix_vector_mult_1D(soln_coeff[istate], soln_at_q[istate],
                                             soln_basis.oneD_vol_operator);
        }
        std::array<std::vector<double>,nstate> entropy_var_at_q;
        for(unsigned int iquad=0; iquad<n_quad_pts; iquad++){
            std::array<double,nstate> soln_state;
            for(int istate=0; istate<nstate; istate++){
                soln_state[istate] = soln_at_q[istate][iquad];
            }

            std::array<double,nstate> entropy_var = this->navier_stokes_physics->convert_conservative_to_entropy(soln_state);
            
            for(int istate=0; istate<nstate; istate++){
                if(iquad==0)
                    entropy_var_at_q[istate].resize(n_quad_pts);
                entropy_var_at_q[istate][iquad] = entropy_var[istate];
            }
        }
        for(int istate=0; istate<nstate; istate++){
            //Projected vector of entropy variables.
            std::vector<double> entropy_var_hat(n_shape_fns);
            vol_projection.matrix_vector_mult_1D(entropy_var_at_q[istate], entropy_var_hat,
                                                 vol_projection.oneD_vol_operator);
                                                
            for(unsigned int ishape=0; ishape<n_shape_fns; ishape++){
                const unsigned int idof = istate * n_shape_fns + ishape;
                entropy_var_hat_global[dofs_indices[idof]] = entropy_var_hat[ishape];
            }
        }
    }

    //Note that dot product accounts for MPI distributed vectors
    //Therefore, there is no need for an MPI sum.
    double entropy = entropy_var_hat_global * mass_matrix_times_solution;
    return entropy;
}

template <int dim, int nstate>
void PeriodicTurbulence<dim, nstate>::compute_unsteady_data_and_write_to_table(
        const unsigned int current_iteration,
        const double current_time,
        const std::shared_ptr <DGBase<dim, double>> dg,
        const std::shared_ptr <dealii::TableHandler> unsteady_data_table)
{
    // Compute and update integrated quantities
    this->compute_and_update_integrated_quantities(*dg);
    // Get computed quantities
    const double integrated_kinetic_energy = this->get_integrated_kinetic_energy();
    const double integrated_enstrophy = this->get_integrated_enstrophy();
    const double vorticity_based_dissipation_rate = this->get_vorticity_based_dissipation_rate();
    const double pressure_dilatation_based_dissipation_rate = this->get_pressure_dilatation_based_dissipation_rate();
    const double deviatoric_strain_rate_tensor_based_dissipation_rate = this->get_deviatoric_strain_rate_tensor_based_dissipation_rate();
    const double strain_rate_tensor_based_dissipation_rate = this->get_strain_rate_tensor_based_dissipation_rate();
    
    double numerical_entropy = 0;
    if (do_calculate_numerical_entropy) numerical_entropy = this->get_numerical_entropy(dg);

    if(this->mpi_rank==0) {
        // Add values to data table
        this->add_value_to_data_table(current_time,"time",unsteady_data_table);
        if(do_calculate_numerical_entropy) this->add_value_to_data_table(numerical_entropy,"numerical_entropy",unsteady_data_table);
        this->add_value_to_data_table(integrated_kinetic_energy,"kinetic_energy",unsteady_data_table);
        this->add_value_to_data_table(integrated_enstrophy,"enstrophy",unsteady_data_table);
        if(is_viscous_flow) this->add_value_to_data_table(vorticity_based_dissipation_rate,"eps_vorticity",unsteady_data_table);
        this->add_value_to_data_table(pressure_dilatation_based_dissipation_rate,"eps_pressure",unsteady_data_table);
        if(is_viscous_flow) this->add_value_to_data_table(strain_rate_tensor_based_dissipation_rate,"eps_strain",unsteady_data_table);
        if(is_viscous_flow) this->add_value_to_data_table(deviatoric_strain_rate_tensor_based_dissipation_rate,"eps_dev_strain",unsteady_data_table);
        // Write to file
        std::ofstream unsteady_data_table_file(this->unsteady_data_table_filename_with_extension);
        unsteady_data_table->write_text(unsteady_data_table_file);
    }
    // Print to console
    this->pcout << "    Iter: " << current_iteration
                << "    Time: " << current_time
                << "    Energy: " << integrated_kinetic_energy
                << "    Enstrophy: " << integrated_enstrophy;
    if(is_viscous_flow) {
        this->pcout << "    eps_vorticity: " << vorticity_based_dissipation_rate
                    << "    eps_p+eps_strain: " << (pressure_dilatation_based_dissipation_rate + strain_rate_tensor_based_dissipation_rate);
    }
    if(do_calculate_numerical_entropy){
        this->pcout << "    Num. Entropy: " << std::setprecision(16) << numerical_entropy;
    }
    this->pcout << std::endl;

    // Abort if energy is nan
    if(std::isnan(integrated_kinetic_energy)) {
        this->pcout << " ERROR: Kinetic energy at time " << current_time << " is nan." << std::endl;
        this->pcout << "        Consider decreasing the time step / CFL number." << std::endl;
        std::abort();
    }

    // Output velocity field for spectra obtaining kinetic energy spectra
    if(output_velocity_field_at_fixed_times) {
        const double time_step = this->get_time_step();
        const double next_time = current_time + time_step;
        const double desired_time = this->output_velocity_field_times[this->index_of_current_desired_time_to_output_velocity_field];
        // Check if current time is an output time
        if((current_time<=desired_time) && (next_time>desired_time)) {
            // Output velocity field for current index
            this->output_velocity_field(dg,this->index_of_current_desired_time_to_output_velocity_field);
            
            // Update index s.t. it never goes out of bounds
            if(this->index_of_current_desired_time_to_output_velocity_field 
                < (this->number_of_times_to_output_velocity_field-1)) {
                this->index_of_current_desired_time_to_output_velocity_field += 1;
            }
        }
    }
}

#if PHILIP_DIM==3
template class PeriodicTurbulence <PHILIP_DIM,PHILIP_DIM+2>;
#endif

} // FlowSolver namespace
} // PHiLiP namespace

