#ifndef __TESTS_H__
#define __TESTS_H__

#include "parameters/all_parameters.h"

#include <deal.II/grid/tria.h>
#include <deal.II/base/conditional_ostream.h>

//#include "dg/dg.h"
namespace PHiLiP {
namespace Tests {

/// Base class of all the tests.
/** Generated by the TestsFactory.
 */
class TestsBase
{
public:
    const Parameters::AllParameters *const all_parameters; ///< Pointer to all parameters
    //const int ndim; ///< Number of dimensions. Run-time variable instead of compile-time constant.
    //const int nstate; ///< Number of state variables. Run-time variable instead of compile-time constant.

    /// Constructor. Deleted the default constructor since it should not be used
    TestsBase () = delete;
    /// Constructor.
    /** @param[in] parameters_input Input parameters.
     */
    TestsBase(const Parameters::AllParameters *const parameters_input);

    /// Destructor.
    virtual ~TestsBase()
    {};

    /// Basically the main and only function of this class.
    /** This will get overloaded by the derived test classes.
     */
    virtual int run_test() const = 0;
protected:
    const MPI_Comm mpi_communicator; ///< MPI communicator.
    const int mpi_rank; ///< MPI rank.
    const int n_mpi; ///< Number of MPI processes.
    /// ConditionalOStream.
    /** Used as std::cout, but only prints if mpi_rank == 0
     */
    dealii::ConditionalOStream pcout;

    /// Evaluates the number of cells to generate the grids for 1D grid based on input file.
    /** @param[in]  ngrids Number of grid sequences to generate.
     *  \return            Vector of 1D grid sizes
     */
    std::vector<int> get_number_1d_cells(const int ngrids) const;

    // /// Evaluates the number of cells to generate the grids for 1D grid based on input file.
    // void globally_refine_and_interpolate(DGBase<dim, double> &dg) const;

};

/// Test factory, that will create the correct test with the right template parameters.
template<int dim, int nstate, typename MeshType = dealii::Triangulation<dim>>
class TestsFactory
{
public:
    /// Recursive factory that will create TestBase<int dim, int nstate>
    /** Must be called with the highest number possible of dimension and nstate. For example
     *
     *  TestBase test = TestFactory::create_test<3,5>(parameters_input)
     */
    /** @param[in] parameters_input Input parameters.
     *  \return                     Smart pointer to the test
     */
    static std::unique_ptr< TestsBase > create_test(const Parameters::AllParameters *const parameters_input);

    /// selects the mesh type to be used in the test
    /** @param[in] parameters_input Input parameters.
     *  \return                     Smart pointer to the test
     */
    static std::unique_ptr< TestsBase > select_mesh(const Parameters::AllParameters *const parameters_input);

    /// Selects the actual test such as grid convergence, numerical flux conversation, etc.
    /** @param[in] parameters_input Input parameters.
     *  \return                     Smart pointer to the test
     */
    static std::unique_ptr< TestsBase > select_test(const Parameters::AllParameters *const parameters_input);
};

} // Tests namespace
} // PHiLiP namespace
#endif

