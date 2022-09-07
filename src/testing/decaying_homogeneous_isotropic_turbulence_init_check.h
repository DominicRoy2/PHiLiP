#ifndef __DECAYING_HOMOGENEOUS_ISOTROPIC_TURBULENCE_INIT_CHECK__
#define __DECAYING_HOMOGENEOUS_ISOTROPIC_TURBULENCE_INIT_CHECK__

#include "tests.h"
#include "dg/dg.h"
#include "parameters/all_parameters.h"

namespace PHiLiP {
namespace Tests {

/// Taylor Green Vortex Restart Check
template <int dim, int nstate>
class DecayingHomogeneousIsotropicTurbulenceInitCheck: public TestsBase
{
public:
    /// Constructor
    DecayingHomogeneousIsotropicTurbulenceInitCheck(
        const Parameters::AllParameters *const parameters_input,
        const dealii::ParameterHandler &parameter_handler_input);

    /// Destructor
    ~DecayingHomogeneousIsotropicTurbulenceInitCheck() {};

    /// Parameter handler for storing the .prm file being ran
    const dealii::ParameterHandler &parameter_handler;
    
    /// Expected kinetic energy at final time
    const double kinetic_energy_expected;

    /// Run test
    int run_test () const override;
protected:
    double compare_solutions(
        DGBase<dim, double> &dg,
        const dealii::LinearAlgebra::distributed::Vector<double> solution_reference,
        const dealii::LinearAlgebra::distributed::Vector<double> solution_to_be_checked) const;

    double integrate_solution_over_domain(
        DGBase<dim, double> &dg,
        const dealii::LinearAlgebra::distributed::Vector<double> solution_input) const;

    /// Renitialize parameters, necessary because parameters created for the test are constant
    Parameters::AllParameters reinit_params(
        const bool output_restart_files_input,
        const bool restart_computation_from_file_input,
        const double final_time_input,
        const double initial_time_input = 0.0,
        const unsigned int initial_iteration_input = 0,
        const double initial_desired_time_for_output_solution_every_dt_time_intervals_input = 0.0,
        const double initial_time_step_input = 0.0,
        const int restart_file_index = 0) const;

    // read the data file
    void read_data_file(
        std::string data_table_filename,
        std::shared_ptr<DGBase<dim,double>> dg) const;
};

} // End of Tests namespace
} // End of PHiLiP namespace

#endif
