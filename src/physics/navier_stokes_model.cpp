#include <cmath>
#include <vector>

#include "ADTypes.hpp"

#include "model.h"
#include "navier_stokes_model.h"

namespace PHiLiP {
namespace Physics {

//================================================================
// Navier Stokes with Model Source Terms Class
//================================================================
template <int dim, int nstate, typename real>
NavierStokesWithModelSourceTerms<dim, nstate, real>::NavierStokesWithModelSourceTerms(
    const double                                              ref_length,
    const double                                              gamma_gas,
    const double                                              mach_inf,
    const double                                              angle_of_attack,
    const double                                              side_slip_angle,
    const double                                              prandtl_number,
    const double                                              reynolds_number_inf,
    const bool                                                use_constant_viscosity,
    const double                                              constant_viscosity,
    const double                                              temperature_inf,
    const double                                              isothermal_wall_temperature,
    const thermal_boundary_condition_enum                     thermal_boundary_condition_type,
    std::shared_ptr< ManufacturedSolutionFunction<dim,real> > manufactured_solution_function,
    const two_point_num_flux_enum                             two_point_num_flux_type)
    : ModelBase<dim,nstate,real>(manufactured_solution_function) 
    , navier_stokes_physics(std::make_unique < NavierStokes<dim,nstate,real> > (
            ref_length,
            gamma_gas,
            mach_inf,
            angle_of_attack,
            side_slip_angle,
            prandtl_number,
            reynolds_number_inf,
            use_constant_viscosity,
            constant_viscosity,
            temperature_inf,
            isothermal_wall_temperature,
            thermal_boundary_condition_type,
            manufactured_solution_function,
            two_point_num_flux_type))
{
    static_assert(nstate==dim+2, "ModelBase::NavierStokesWithModelSourceTerms() should be created with nstate=dim+2");
    // initialize zero arrays / tensors
    for (int s=0; s<nstate; ++s) {
        zero_array[s] = 0.0;
        for (int d=0; d<dim; ++d) {
            zero_tensor_array[s][d] = 0.0;
        }
    }
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::convective_flux (
    const std::array<real,nstate> &/*conservative_soln*/) const
{
    return this->zero_tensor_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<dealii::Tensor<1,dim,real>,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::dissipative_flux (
    const std::array<real,nstate> &/*conservative_soln*/,
    const std::array<dealii::Tensor<1,dim,real>,nstate> &/*solution_gradient*/,
    const dealii::types::global_dof_index /*cell_index*/) const
{   
    return this->zero_tensor_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::source_term (
        const dealii::Point<dim,real> &/*pos*/,
        const std::array<real,nstate> &/*solution*/,
        const real /*current_time*/,
        const dealii::types::global_dof_index /*cell_index*/) const
{
    return this->zero_array;
}
//----------------------------------------------------------------
template <int dim, int nstate, typename real>
std::array<real,nstate> NavierStokesWithModelSourceTerms<dim,nstate,real>
::channel_flow_source_term (
    const std::array<real,nstate> &conservative_soln) const
{
    std::array<real,nstate> source_term;
    std::fill(source_term.begin(), source_term.end(), 0.0);

    // get bulk velocity
    const real density = conservative_soln[0];
    const std::array<real,nstate> primitive_soln = this->navier_stokes_physics->convert_conservative_to_primitive(conservative_soln);
    const real viscosity_coefficient = this->navier_stokes_physics->compute_viscosity_coefficient(primitive_soln);
    const real bulk_velocity = viscosity_coefficient*this->channel_bulk_reynolds_number/(density*this->half_channel_height);
    
    // x-momentum term
    source_term[1] = (this->integrated_density_over_domain*bulk_velocity - conservative_soln[1])/this->time_step;
    
    // energy term
    const real x_velocity = primitive_soln[1];
    source_term[nstate-1] = x_velocity*source_term[1];
    
    return source_term;
}
//----------------------------------------------------------------
//----------------------------------------------------------------
//----------------------------------------------------------------
// Instantiate explicitly
// -- NavierStokesWithModelSourceTerms
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, double >;
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, FadType  >;
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, RadType  >;
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, FadFadType >;
template class NavierStokesWithModelSourceTerms< PHILIP_DIM, PHILIP_DIM+2, RadFadType >;

} // Physics namespace
} // PHiLiP namespace