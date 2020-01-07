#include <array>

#include "parameters_grid_refinement_study.h"

#include "parameters/parameters_manufactured_convergence_study.h"

namespace PHiLiP {

namespace Parameters {

GridRefinementStudyParam::GridRefinementStudyParam() : 
    functional_param(FunctionalParam()),
    manufactured_solution_param(ManufacturedSolutionParam()) {}

void GridRefinementStudyParam::declare_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("grid refinement study");
    {
        Parameters::FunctionalParam::declare_parameters(prm);
        Parameters::ManufacturedSolutionParam::declare_parameters(prm);

        prm.enter_subsection("grid refinement");
        {
            Parameters::GridRefinementParam::declare_parameters(prm);
        }
        prm.leave_subsection();

        // looping over the elements of the array to declare the different refinement
        for(unsigned int i = 0; i < MAX_REFINEMENTS; ++i)
        {
            prm.enter_subsection("grid refinement [" + dealii::Utilities::int_to_string(i,1) + "]");
            {
                Parameters::GridRefinementParam::declare_parameters(prm);
            }
            prm.leave_subsection();
        }
    
        prm.declare_entry("poly_degree", "1",
                          dealii::Patterns::Integer(),
                          "Polynomial order of starting mesh.");

        prm.declare_entry("poly_degree_max", "5",
                          dealii::Patterns::Integer(),
                          "Maximum polynomial order.");

        prm.declare_entry("poly_degree_grid", "2",
                          dealii::Patterns::Integer(),
                          "Polynomial degree of the grid.");

        prm.declare_entry("num_refinements", "0",
                          dealii::Patterns::Integer(0, MAX_REFINEMENTS),
                          "Number of different refinements to be performed.");

        prm.declare_entry("grid_type", "hypercube",
                          dealii::Patterns::Selection("hypercube|sinehypercube|read_grid"),
                          "Enum of generated grid. "
                          "If read_grid, must have grids xxxx.msh."
                          "Choices are <hypercube|read_grid>.");

        prm.declare_entry("input_grid", "xxxx",
                          dealii::Patterns::FileName(dealii::Patterns::FileName::FileType::input),
                          "Name of Gmsh grid xxxx.msh used in the grid refinement study if read_grid is chosen as the grid_type.");

        prm.declare_entry("grid_left", "0.0",
                          dealii::Patterns::Double(),
                          "for grid_type hypercube, left bound of domain.");

        prm.declare_entry("grid_right", "1.0",
                          dealii::Patterns::Double(),
                          "for grid_type hypercube, right bound of domain.");

        prm.declare_entry("grid_size", "4",
                          dealii::Patterns::Integer(),
                          "Initial grid size (number of elements per side).");
    }
    prm.leave_subsection();
}

void GridRefinementStudyParam::parse_parameters(dealii::ParameterHandler &prm)
{
    prm.enter_subsection("grid refinement study");
    {
        functional_param.parse_parameters(prm);
        manufactured_solution_param.parse_parameters(prm);

        // default section gets written into vector pos 1 by default
        prm.enter_subsection("grid refinement");
        {
            grid_refinement_param_vector[0].parse_parameters(prm);
        }
        prm.leave_subsection();

        num_refinements = prm.get_integer("num_refinements");

        // overwrite if num_refinements > 0
        for(unsigned int i = 0; i < num_refinements; ++i)
        {
            prm.enter_subsection("grid refinement [" + dealii::Utilities::int_to_string(i,1) + "]");
            {
                grid_refinement_param_vector[i].parse_parameters(prm);
            }
            prm.leave_subsection();
        }

        poly_degree      = prm.get_integer("poly_degree");
        poly_degree_max  = prm.get_integer("poly_degree_max");
        poly_degree_grid = prm.get_integer("poly_degree_grid");

        const std::string grid_string = prm.get("grid_type");
        using GridEnum = Parameters::ManufacturedConvergenceStudyParam::GridEnum;
        if(grid_string == "hypercube")      {grid_type = GridEnum::hypercube;}
        else if(grid_string == "read_grid") {grid_type = GridEnum::read_grid;
                                             input_grid = prm.get("input_grid");}

        grid_left  = prm.get_double("grid_left");
        grid_right = prm.get_double("grid_right");

        grid_size  = prm.get_integer("grid_size");
    }
    prm.leave_subsection();
}

} // namespace Parameters

} // namespace PHiLiP