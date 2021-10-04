#include <iostream>

#include "reduced_order.h"
#include "parameters/all_parameters.h"
#include "pod/proper_orthogonal_decomposition.h"
#include <fstream>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include "burgers_rewienski_snapshot.h"
#include "parameters/all_parameters.h"
#include "dg/dg_factory.hpp"
#include "ode_solver/ode_solver_factory.h"
#include "ode_solver/pod_galerkin_ode_solver.h"


namespace PHiLiP {
namespace Tests {

template <int dim, int nstate>
ReducedOrder<dim, nstate>::ReducedOrder(const PHiLiP::Parameters::AllParameters *const parameters_input)
        : TestsBase::TestsBase(parameters_input)
{}

template <int dim, int nstate>
int ReducedOrder<dim, nstate>::run_test() const
{
    int num_basis = 30;

    std::shared_ptr<ProperOrthogonalDecomposition::POD> pod= std::make_shared<ProperOrthogonalDecomposition::POD>(num_basis);

    std::ofstream out_file("full_U_matrix.txt");
    pod->get_full_pod_basis().print_formatted(out_file, 4);
    out_file.close();

    std::ofstream out_file2("pod_basis_matrix.txt");
    pod->build_reduced_pod_basis();
    pod->pod_basis.print(out_file2);
    out_file2.close();

    const Parameters::AllParameters param = *(TestsBase::all_parameters);

    pcout << "Running Burgers Rewienski with parameter a: "
          << param.reduced_order_param.rewienski_a
          << " and parameter b: "
          << param.reduced_order_param.rewienski_b
          << std::endl;

    using Triangulation = dealii::Triangulation<dim>;
    std::shared_ptr<Triangulation> grid = std::make_shared<Triangulation>();

    double left = param.grid_refinement_study_param.grid_left;
    double right = param.grid_refinement_study_param.grid_right;
    const bool colorize = true;
    int n_refinements = param.grid_refinement_study_param.num_refinements;
    unsigned int poly_degree = param.grid_refinement_study_param.poly_degree;
    dealii::GridGenerator::hyper_cube(*grid, left, right, colorize);

    grid->refine_global(n_refinements);
    pcout << "Grid generated and refined" << std::endl;

    std::shared_ptr < PHiLiP::DGBase<dim, double> > dg = PHiLiP::DGFactory<dim,double>::create_discontinuous_galerkin(all_parameters, poly_degree, grid);
    pcout << "dg created" <<std::endl;
    dg->allocate_system ();

    pcout << "Implement initial conditions" << std::endl;
    dealii::FunctionParser<1> initial_condition;
    std::string variables = "x";
    std::map<std::string,double> constants;
    constants["pi"] = dealii::numbers::PI;
    std::string expression = "1";
    initial_condition.initialize(variables, expression, constants);
    dealii::VectorTools::interpolate(dg->dof_handler,initial_condition,dg->solution);

    pcout << "Create ODE solver" << std::endl;
    // Create ODE solver using the factory and providing the DG object
    std::shared_ptr<PHiLiP::ODE::ODESolverBase<dim, double>> ode_solver = PHiLiP::ODE::ODESolverFactory<dim, double>::create_ODESolver(dg, pod);

    pcout << "Advancing solution time" << std::endl;

    double finalTime = param.reduced_order_param.final_time;
    ode_solver->advance_solution_time(finalTime);

    return 0;
}
#if PHILIP_DIM==1
        template class ReducedOrder<PHILIP_DIM,PHILIP_DIM>;
#endif
    } // Tests namespace
} // PHiLiP namespace
