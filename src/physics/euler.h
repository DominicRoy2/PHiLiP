#ifndef __EULER__
#define __EULER__

#include <deal.II/base/tensor.h>
#include "physics.h"
#include "parameters/all_parameters.h"
#include "parameters/parameters_manufactured_solution.h"

namespace PHiLiP {
namespace Physics {

/// Euler equations. Derived from PhysicsBase
/** Only 2D and 3D
 *  State variable and convective fluxes given by
 *
 *  \f[ 
 *  \mathbf{w} = 
 *  \begin{bmatrix} \rho \\ \rho v_1 \\ \rho v_2 \\ \rho v_3 \\ \rho E \end{bmatrix}
 *  , \qquad
 *  \mathbf{F}_{conv} = 
 *  \begin{bmatrix} 
 *      \mathbf{f}^x_{conv}, \mathbf{f}^y_{conv}, \mathbf{f}^z_{conv}
 *  \end{bmatrix}
 *  =
 *  \begin{bmatrix} 
 *  \begin{bmatrix} 
 *  \rho v_1 \\
 *  \rho v_1 v_1 + p \\
 *  \rho v_1 v_2     \\ 
 *  \rho v_1 v_3     \\
 *  v_1 (\rho e+p)
 *  \end{bmatrix}
 *  ,
 *  \begin{bmatrix} 
 *  \rho v_2 \\
 *  \rho v_1 v_2     \\
 *  \rho v_2 v_2 + p \\ 
 *  \rho v_2 v_3     \\
 *  v_2 (\rho e+p)
 *  \end{bmatrix}
 *  ,
 *  \begin{bmatrix} 
 *  \rho v_3 \\
 *  \rho v_1 v_3     \\
 *  \rho v_2 v_3     \\ 
 *  \rho v_3 v_3 + p \\
 *  v_3 (\rho e+p)
 *  \end{bmatrix}
 *  \end{bmatrix} \f]
 *  
 *  where, \f$ E \f$ is the specific total energy and \f$ e \f$ is the specific internal
 *  energy, related by
 *  \f[
 *      E = e + |V|^2 / 2
 *  \f] 
 *  For a calorically perfect gas
 *
 *  \f[
 *  p=(\gamma -1)(\rho e-\frac{1}{2}\rho \|\mathbf{v}\|)
 *  \f]
 *
 *  Dissipative flux \f$ \mathbf{F}_{diss} = \mathbf{0} \f$
 *
 *  Source term \f$ s(\mathbf{x}) \f$
 *
 *  Equation:
 *  \f[ \boldsymbol{\nabla} \cdot
 *         (  \mathbf{F}_{conv}( w ) 
 *          + \mathbf{F}_{diss}( w, \boldsymbol{\nabla}(w) )
 *      = s(\mathbf{x})
 *  \f]
 *
 *
 *  Still need to provide functions to un-non-dimensionalize the variables.
 *  Like, given density_inf
 */
template <int dim, int nspecies, int nstate, typename real>
class Euler : public PhysicsBase <dim, nspecies, nstate, real>
{
protected:
    // For overloading the virtual functions defined in PhysicsBase
    /** Once you overload a function from Base class in Derived class,
     *  all functions with the same name in the Base class get hidden in Derived class.  
     *  
     *  Solution: In order to make the hidden function visible in derived class, 
     *  we need to add the following: */
    using PhysicsBase<dim,nspecies,nstate,real>::dissipative_flux;
    using PhysicsBase<dim,nspecies,nstate,real>::source_term;
    using PhysicsBase<dim,nspecies,nstate,real>::boundary_face_values;
public:
    using two_point_num_flux_enum = Parameters::AllParameters::TwoPointNumericalFlux;
    /// Constructor
    Euler ( 
        const Parameters::AllParameters *const                    parameters_input,
        const double                                              ref_length,
        const double                                              gamma_gas,
        const double                                              mach_inf,
        const double                                              angle_of_attack,
        const double                                              side_slip_angle,
        std::shared_ptr< ManufacturedSolutionFunction<dim,nspecies,real> > manufactured_solution_function = nullptr,
        const two_point_num_flux_enum                             two_point_num_flux_type = two_point_num_flux_enum::KG,
        const bool                                                has_nonzero_diffusion = false,
        const bool                                                has_nonzero_physical_source = false);

    const double ref_length; ///< Reference length.
    const double gam; ///< Constant heat capacity ratio of fluid.
    const double gamm1; ///< Constant heat capacity ratio (Gamma-1.0) used often.

    /// Non-dimensionalized density* at infinity. density* = density/density_ref
    /// Choose density_ref = density(inf)
    /// density*(inf) = density(inf) / density_ref = density(inf)/density(inf) = 1.0
    const double density_inf;

    const double mach_inf; ///< Farfield Mach number.
    const double mach_inf_sqr; ///< Farfield Mach number squared.
    /// Angle of attack.
    /** Mandatory for 2D simulations.
     */
    const double angle_of_attack;
    /// Sideslip angle.
    /** Mandatory for 2D and 3D simulations.
     */
    const double side_slip_angle;


    const double sound_inf; ///< Non-dimensionalized sound* at infinity
    const double pressure_inf; ///< Non-dimensionalized pressure* at infinity
    const double entropy_inf; ///< Entropy measure at infinity
    const two_point_num_flux_enum two_point_num_flux_type; ///< Two point numerical flux type (for split form)
    double temperature_inf; ///< Non-dimensionalized temperature* at infinity. Should equal 1/density*(inf)
    double dynamic_pressure_inf; ///< Non-dimensionalized dynamic pressure* at infinity

    //const double internal_energy_inf;
    /// Non-dimensionalized Velocity vector at farfield
    /** Evaluated using mach_number, angle_of_attack, and side_slip_angle.
     */
    dealii::Tensor<1,dim,double> velocities_inf; // should be const


    // dealii::Tensor<1,dim,double> compute_velocities_inf() const;

    // std::array<real,nstate> manufactured_solution (const dealii::Point<dim,double> &pos) const;

    /// Convective flux: \f$ \mathbf{F}_{conv} \f$
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_flux (
        const std::array<real,nstate> &conservative_soln) const override;

    /// Convective normal flux: \f$ \mathbf{F}_{conv} \cdot \hat{n} \f$
    std::array<real,nstate> convective_normal_flux (const std::array<real,nstate> &conservative_soln, const dealii::Tensor<1,dim,real> &normal) const;

    /// Convective flux Jacobian: \f$ \frac{\partial \mathbf{F}_{conv}}{\partial w} \cdot \mathbf{n} \f$
    dealii::Tensor<2,nstate,real> convective_flux_directional_jacobian (
        const std::array<real,nstate> &conservative_soln,
        const dealii::Tensor<1,dim,real> &normal) const;

    /// Spectral radius of convective term Jacobian is 'c'
    std::array<real,nstate> convective_eigenvalues (
        const std::array<real,nstate> &/*conservative_soln*/,
        const dealii::Tensor<1,dim,real> &/*normal*/) const override;

    /// Maximum convective eigenvalue
    real max_convective_eigenvalue (const std::array<real,nstate> &soln) const override;

    /// Maximum convective normal eigenvalue (used in Lax-Friedrichs)
    /** See the book I do like CFD, equation 3.6.18 */
    real max_convective_normal_eigenvalue (
        const std::array<real,nstate> &soln,
        const dealii::Tensor<1,dim,real> &normal) const override;

    /// Maximum viscous eigenvalue.
    real max_viscous_eigenvalue (const std::array<real,nstate> &soln) const override;

    /// Dissipative flux: 0
    std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient,
        const dealii::types::global_dof_index cell_index) const;

    /// (function overload) Dissipative flux: 0
    virtual std::array<dealii::Tensor<1,dim,real>,nstate> dissipative_flux (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &solution_gradient) const;

    /// Source term is zero or depends on manufactured solution
    std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real current_time,
        const dealii::types::global_dof_index cell_index) const;

    /// (function overload) Source term is zero or depends on manufactured solution
    virtual std::array<real,nstate> source_term (
        const dealii::Point<dim,real> &pos,
        const std::array<real,nstate> &conservative_soln,
        const real current_time) const;

    /// Convective flux contribution to the source term
    std::array<real,nstate> convective_source_term (
        const dealii::Point<dim,real> &pos) const;

protected:
    /// Check positive quantity and modify it according to handle_non_physical_result()
    /** in PhysicsBase class
     */
    template<typename real2>
    bool check_positive_quantity(real2 &quantity, const std::string qty_name) const;

public:
    /// Given conservative variables [density, [momentum], total energy],
    /// returns primitive variables [density, [velocities], pressure].
    ///
    /// Opposite of convert_primitive_to_conservative
    template<typename real2>
    std::array<real2,nstate> convert_conservative_to_primitive_templated ( const std::array<real2,nstate> &conservative_soln ) const;

    /// Convert conservative to primitive (real2==real); required by base class
    std::array<real,nstate> convert_conservative_to_primitive ( const std::array<real,nstate> &conservative_soln ) const;

    /// Given primitive variables [density, [velocities], pressure],
    /// returns conservative variables [density, [momentum], total energy].
    ///
    /// Opposite of convert_primitive_to_conservative
    std::array<real,nstate> convert_primitive_to_conservative ( const std::array<real,nstate> &primitive_soln ) const;

    /** Obtain gradient of primitive variables from gradient of conservative variables */
    template<typename real2>
    std::array<dealii::Tensor<1,dim,real2>,nstate> 
    convert_conservative_gradient_to_primitive_gradient_templated (
        const std::array<real2,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real2>,nstate> &conservative_soln_gradient) const;

    /** Obtain gradient of primitive variables from gradient of conservative variables */
    std::array<dealii::Tensor<1,dim,real>,nstate> 
    convert_conservative_gradient_to_primitive_gradient (
        const std::array<real,nstate> &conservative_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &conservative_soln_gradient) const;

    /** Obtain gradient of conservative variables from gradient of primitive variables */
    std::array<dealii::Tensor<1,dim,real>,nstate> 
    convert_primitive_gradient_to_conservative_gradient (
        const std::array<real,nstate> &primitive_soln,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &primitive_soln_gradient) const;

    /// Evaluate pressure from conservative variables
    template<typename real2>
    real2 compute_pressure_templated ( const std::array<real2,nstate> &conservative_soln ) const;

    real compute_pressure ( const std::array<real,nstate> &conservative_soln ) const;

    /// Evaluate physical entropy = log(p \rho^{-\gamma}) from pressure and density
    template<typename real2>
    real2 compute_entropy (const real2 density, const real2 pressure) const;

    /// Evaluate pressure from conservative variables
    real compute_specific_enthalpy ( const std::array<real,nstate> &conservative_soln, const real pressure) const;

    /// Compute numerical entropy function -rho s 
    real compute_numerical_entropy_function(const std::array<real,nstate> &conservative_soln) const;

    /// Evaluate speed of sound from conservative variables
    real compute_sound ( const std::array<real,nstate> &conservative_soln ) const;
    /// Evaluate speed of sound from density and pressure
    real compute_sound ( const real density, const real pressure ) const;

    /// Evaluate velocities from conservative variables
    template<typename real2>
    dealii::Tensor<1,dim,real2> compute_velocities ( const std::array<real2,nstate> &conservative_soln ) const;
    /// Given the velocity vector \f$ \mathbf{u} \f$, returns the dot-product  \f$ \mathbf{u} \cdot \mathbf{u} \f$
    template<typename real2>
    real2 compute_velocity_squared ( const dealii::Tensor<1,dim,real2> &velocities ) const;

    /// Given primitive variables, returns velocities.
    template<typename real2>
    dealii::Tensor<1,dim,real2> extract_velocities_from_primitive ( const std::array<real2,nstate> &primitive_soln ) const;
    /// Given primitive variables, returns total energy
    /** @param[in] primitive_soln    Primitive solution (density, momentum, energy)
     *  \return                      Entropy measure
     */
    real compute_total_energy ( const std::array<real,nstate> &primitive_soln ) const;

    /// Given primitive variables, returns kinetic energy
    real compute_kinetic_energy_from_primitive_solution ( const std::array<real,nstate> &primitive_soln ) const;

    /// Given primitive variables, returns incompressible kinetic energy
    real compute_incompressible_kinetic_energy_from_primitive_solution ( const std::array<real,nstate> &primitive_soln ) const;

    /// Given conservative variables, returns kinetic energy
    real compute_kinetic_energy_from_conservative_solution ( const std::array<real,nstate> &conservative_soln ) const;

    /// Given conservative variables, returns incompressible kinetic energy
    real compute_incompressible_kinetic_energy_from_conservative_solution ( const std::array<real,nstate> &conservative_soln ) const;

    /// Evaluate entropy from conservative variables
    /** Note that it is not the actual entropy since it's missing some constants.
     *  Used to check entropy convergence
     *  See discussion in
     *  https://physics.stackexchange.com/questions/116779/entropy-is-constant-how-to-express-this-equation-in-terms-of-pressure-and-densi?answertab=votes#tab-top
     *
     *  @param[in] conservative_soln Conservative solution (density, momentum, energy)
     *  \return                      Entropy measure
     */
    real compute_entropy_measure ( const std::array<real,nstate> &conservative_soln ) const;

    /// Evaluate entropy from density and pressure. 
    real compute_entropy_measure ( const real density, const real pressure ) const;

    /// Given conservative variables, returns Mach number
    real compute_mach_number ( const std::array<real,nstate> &conservative_soln ) const;

    /// Given primitive variables, returns NON-DIMENSIONALIZED temperature using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    template<typename real2>
    real2 compute_temperature ( const std::array<real2,nstate> &primitive_soln ) const;

    /// Given pressure and temperature, returns NON-DIMENSIONALIZED density using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_density_from_pressure_temperature ( const real pressure, const real temperature ) const;

    /// Given density and pressure, returns NON-DIMENSIONALIZED temperature using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_temperature_from_density_pressure ( const real density, const real pressure ) const;

    /// Given density and temperature, returns NON-DIMENSIONALIZED pressure using free-stream non-dimensionalization
    /** See the book I do like CFD, sec 4.14.2 */
    real compute_pressure_from_density_temperature ( const real density, const real temperature ) const;

    ///  Evaluates convective flux based on the chosen split form.
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const override;

    /// Computes the entropy variables.
    /// Given conservative variables [density, [momentum], total energy],
    /// Computes entropy variables according to Chan 2018, eq. 119
    std::array<real,nstate> compute_entropy_variables (
                const std::array<real,nstate> &conservative_soln) const;

    /// Computes the conservative variables [density, [momentum], total energy
    /// from the entropy variables according to Chan 2018, eq. 120
    std::array<real,nstate> compute_conservative_variables_from_entropy_variables (
                const std::array<real,nstate> &entropy_var) const;

    /// Computes the kinetic energy variables.
    std::array<real,nstate> compute_kinetic_energy_variables (
                const std::array<real,nstate> &conservative_soln) const;

    /// Mean density given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    real compute_mean_density(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    /// Mean pressure given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    real compute_mean_pressure(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    /// Mean velocities given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    dealii::Tensor<1,dim,real> compute_mean_velocities(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    /// Mean specific total energy given two sets of conservative solutions.
    /** Used in the implementation of the split form.
     */
    real compute_mean_specific_total_energy(
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &convervative_soln2) const;

    /// Boundary condition handler
    void boundary_face_values (
        const int /*boundary_type*/,
        const dealii::Point<dim, real> &/*pos*/,
        const dealii::Tensor<1,dim,real> &/*normal*/,
        const std::array<real,nstate> &/*soln_int*/,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_int*/,
        std::array<real,nstate> &/*soln_bc*/,
        std::array<dealii::Tensor<1,dim,real>,nstate> &/*soln_grad_bc*/) const;

    /// For post processing purposes, computes all the quantities we write to the VTK files
    virtual dealii::Vector<double> post_compute_derived_quantities_vector (
        const dealii::Vector<double>              &uh,
        const std::vector<dealii::Tensor<1,dim> > &duh,
        const std::vector<dealii::Tensor<2,dim> > &dduh,
        const dealii::Tensor<1,dim>               &normals,
        const dealii::Point<dim>                  &evaluation_points) const override;
    
    /// For post processing purposes, sets the base names (with no prefix or suffix) of the computed quantities
    virtual std::vector<std::string> post_get_names () const override;
    
    /// For post processing purposes, sets the interpretation of each computed quantity as either scalar or vector
    virtual std::vector<dealii::DataComponentInterpretation::DataComponentInterpretation> post_get_data_component_interpretation () const override;
    
    /// For post processing purposes, updates the required flags for dealii
    virtual dealii::UpdateFlags post_get_needed_update_flags () const override;
    
    real compute_U_plus_from_DNS(const real y_plus) const;

    std::vector<real> y_plus_values = {0, 0.0356349, 0.1425382, 0.3207058, 0.5701312, 0.8908048, 1.2827147, 1.745846, 2.2801813, 2.8857006, 3.5623809, 4.3101969, 5.1291203, 6.0191204, 6.9801637, 8.0122139, 9.1152321, 10.289177, 11.5340042, 12.8496669, 14.2361156, 15.6932981, 17.2211595, 18.8196422, 20.4886862, 22.2282285, 24.0382037, 25.9185437, 27.8691775, 29.8900319, 31.9810307, 34.1420951, 36.3731439, 38.674093, 41.0448558, 43.485343, 45.9954628, 48.5751206, 51.2242194, 53.9426593, 56.7303381, 59.5871508, 62.5129898, 65.507745, 68.5713036, 71.7035503, 74.9043671, 78.1736336, 81.5112266, 84.9170206, 88.3908872, 91.9326957, 95.5423128, 99.2196024, 102.9644263, 106.7766434, 110.6561101, 114.6026805, 118.6162059, 122.6965352, 126.8435148, 131.0569886, 135.3367978, 139.6827815, 144.094776, 148.5726152, 153.1161304, 157.7251506, 162.3995023, 167.1390095, 171.9434938, 176.8127743, 181.7466676, 186.744988, 191.8075473, 196.934155, 202.1246179, 207.3787407, 212.6963255, 218.0771722, 223.5210782, 229.0278385, 234.5972458, 240.2290903, 245.9231602, 251.6792409, 257.4971158, 263.3765658, 269.3173696, 275.3193035, 281.3821415, 287.5056554, 293.6896147, 299.9337864, 306.2379356, 312.6018248, 319.0252145, 325.5078628, 332.0495257, 338.6499569, 345.3089078, 352.0261278, 358.801364, 365.6343613, 372.5248623, 379.4726078, 386.4773361, 393.5387836, 400.6566842, 407.8307701, 415.0607712, 422.3464152, 429.6874278, 437.0835328, 444.5344515, 452.0399035, 459.5996062, 467.213275, 474.8806232, 482.6013622, 490.3752013, 498.2018478, 506.081007, 514.0123823, 521.9956751, 530.0305848, 538.1168089, 546.254043, 554.4419806, 562.6803136, 570.9687318, 579.306923, 587.6945734, 596.1313672, 604.6169867, 613.1511125, 621.7334233, 630.3635959, 639.0413054, 647.7662252, 656.5380266, 665.3563795, 674.2209519, 683.13141, 692.0874184, 701.0886397, 710.1347353, 719.2253644, 728.3601849, 737.5388527, 746.7610224, 756.0263467, 765.3344768, 774.6850623, 784.077751, 793.5121894, 802.9880223, 812.5048929, 822.0624429, 831.6603124, 841.2981402, 850.9755633, 860.6922174, 870.4477367, 880.2417539, 890.0739002, 899.9438055, 909.8510982, 919.7954053, 929.7763523, 939.7935635, 949.8466618, 959.9352686, 970.0590042, 980.2174874, 990.4103357, 1000.6371653, 1010.8975913, 1021.1912273, 1031.5176857, 1041.8765779, 1052.2675138, 1062.6901021, 1073.1439505, 1083.6286654, 1094.143852, 1104.6891145, 1115.2640559, 1125.8682779, 1136.5013813, 1147.1629659, 1157.8526302, 1168.5699718, 1179.3145871, 1190.0860716, 1200.8840198, 1211.7080252, 1222.5576802, 1233.4325763, 1244.3323042, 1255.2564534, 1266.2046126, 1277.1763698, 1288.1713117, 1299.1890244, 1310.2290931, 1321.2911023, 1332.3746353, 1343.4792749, 1354.604603, 1365.7502008, 1376.9156486, 1388.1005261, 1399.3044121, 1410.5268849, 1421.7675218, 1433.0258998, 1444.3015949, 1455.5941826, 1466.9032378, 1478.2283347, 1489.5690468, 1500.9249473, 1512.2956085, 1523.6806025, 1535.0795004, 1546.4918733, 1557.9172913, 1569.3553244, 1580.8055419, 1592.2675127, 1603.7408053, 1615.2249877, 1626.7196276, 1638.2242921, 1649.7385482, 1661.2619623, 1672.7941006, 1684.3345289, 1695.8828126, 1707.4385172, 1719.0012073, 1730.5704478, 1742.1458031, 1753.7268373, 1765.3131144, 1776.9041983, 1788.4996524, 1800.0990403, 1811.7019253, 1823.3078704, 1834.9164388, 1846.5271934, 1858.139697, 1869.7535124, 1881.3682025, 1892.9833298, 1904.5984572, 1916.2131473, 1927.8269627, 1939.4394663, 1951.0502209, 1962.6587893, 1974.2647344, 1985.8676194, 1997.4670073, 2009.0624614, 2020.6535453, 2032.2398224, 2043.8208566, 2055.3962119, 2066.9654524, 2078.5281425, 2090.083847, 2101.6321308, 2113.1725591, 2124.7046974, 2136.2281115, 2147.7423676, 2159.2470321, 2170.741672, 2182.2258544, 2193.699147, 2205.1611178, 2216.6113353, 2228.0493684, 2239.4747864, 2250.8871593, 2262.2860572, 2273.6710512, 2285.0417124, 2296.3976129, 2307.738325, 2319.0634219, 2330.372477, 2341.6650648, 2352.9407599, 2364.1991379, 2375.4397748, 2386.6622476, 2397.8661336, 2409.0510111, 2420.2164589, 2431.3620567, 2442.4873848, 2453.5920244, 2464.6755574, 2475.7375665, 2486.7776353, 2497.795348, 2508.7902899, 2519.7620471, 2530.7102063, 2541.6343555, 2552.5340834, 2563.4089795, 2574.2586345, 2585.0826399, 2595.8805881, 2606.6520726, 2617.3966879, 2628.1140295, 2638.8036938, 2649.4652784, 2660.0983818, 2670.7026038, 2681.2775452, 2691.8228077, 2702.3379943, 2712.8227092, 2723.2765576, 2733.6991459, 2744.0900818, 2754.448974, 2764.7754324, 2775.0690684, 2785.3294944, 2795.556324, 2805.7491723, 2815.9076555, 2826.031391, 2836.1199979, 2846.1730962, 2856.1903074, 2866.1712544, 2876.1155615, 2886.0228542, 2895.8927595, 2905.7249058, 2915.518923, 2925.2744423, 2934.9910964, 2944.6685195, 2954.3063473, 2963.9042168, 2973.4617668, 2982.9786374, 2992.4544703, 3001.8889087, 3011.2815974, 3020.6321829, 3029.940313, 3039.2056373, 3048.427807, 3057.6064748, 3066.7412953, 3075.8319244, 3084.8780199, 3093.8792413, 3102.8352497, 3111.7457078, 3120.6102801, 3129.4286331, 3138.2004345, 3146.9253543, 3155.6030638, 3164.2332364, 3172.8155471, 3181.3496729, 3189.8352925, 3198.2720863, 3206.6597367, 3214.9979279, 3223.2863461, 3231.5246791, 3239.7126167, 3247.8498508, 3255.9360749, 3263.9709846, 3271.9542774, 3279.8856527, 3287.7648119, 3295.5914584, 3303.3652975, 3311.0860365, 3318.7533847, 3326.3670535, 3333.9267562, 3341.4322082, 3348.8831269, 3356.2792319, 3363.6202445, 3370.9058885, 3378.1358896, 3385.3099755, 3392.4278761, 3399.4893235, 3406.4940519, 3413.4417973, 3420.3322984, 3427.1652957, 3433.9405319, 3440.6577519, 3447.3167028, 3453.917134, 3460.4587969, 3466.9414452, 3473.3648349, 3479.7287241, 3486.0328733, 3492.277045, 3498.4610043, 3504.5845182, 3510.6473562, 3516.6492901, 3522.5900939, 3528.4695439, 3534.2874188, 3540.0434995, 3545.7375694, 3551.3694139, 3556.9388212, 3562.4455815, 3567.8894875, 3573.2703342, 3578.587919, 3583.8420418, 3589.0325047, 3594.1591124, 3599.2216717, 3604.2199921, 3609.1538854, 3614.0231659, 3618.8276502, 3623.5671574, 3628.2415091, 3632.8505293, 3637.3940445, 3641.8718837, 3646.2838781, 3650.6298618, 3654.9096711, 3659.1231449, 3663.2701245, 3667.3504538, 3671.3639792, 3675.3105496, 3679.1900163, 3683.0022334, 3686.7470572, 3690.4243469, 3694.033964, 3697.5757725, 3701.0496391, 3704.455433, 3707.7930261, 3711.0622926, 3714.2631094, 3717.3953561, 3720.4589147, 3723.4536699, 3726.3795089, 3729.2363216, 3732.0240004, 3734.7424403, 3737.3915391, 3739.9711969, 3742.4813167, 3744.9218039, 3747.2925667, 3749.5935158, 3751.8245646, 3753.985629, 3756.0766278, 3758.0974822, 3760.048116, 3761.928456, 3763.7384312, 3765.4779735, 3767.1470174, 3768.7455002, 3770.2733616, 3771.7305441, 3773.1169928, 3774.4326555, 3775.6774827, 3776.8514276, 3777.9544458, 3778.986496, 3779.9475393, 3780.8375394, 3781.6564628, 3782.4042788, 3783.0809591, 3783.6864784, 3784.2208137, 3784.683945, 3785.0758549, 3785.3965285, 3785.6459539, 3785.8241215, 3785.9310248, 3785.9666597};
    
    std::vector<real> u_plus_values = {0, 0.0356349, 0.142538, 0.3207028, 0.5701014, 0.8906251, 1.2819293, 1.7431135, 2.2721801, 2.8652833, 
        3.5159183, 4.2143387, 4.9475332, 5.6999485, 6.4548765, 7.1961624, 7.9097843, 8.5849414, 9.2144896, 9.7947803, 10.3250913, 10.8068788,
        11.2430314, 11.6372416, 11.9935342, 12.3159472, 12.6083372, 12.8742741, 13.1169986, 13.3394193, 13.5441279, 13.7334253, 13.9093498, 
        14.0736992, 14.2280539, 14.3738023, 14.5121564, 14.6441606, 14.770708, 14.8925672, 15.0103865, 15.1247022, 15.2359588, 15.3445344, 
        15.4507665, 15.5549486, 15.6573296, 15.7581197, 15.8574849, 15.9555505, 16.0524232, 16.1482094, 16.2430194, 16.3369752, 16.4301871, 
        16.5227238, 16.6146373, 16.7059929, 16.7968663, 16.8873377, 16.9774809, 17.0673609, 17.1570217, 17.2464879, 17.3357671, 17.4248629, 
        17.5138161, 17.602695, 17.6915476, 17.7804324, 17.8693872, 17.9584219, 18.047513, 18.1366333, 18.2257701, 18.3149191, 18.4040871, 
        18.4932899, 18.582534, 18.6718024, 18.7610664, 18.8502683, 18.9393954, 19.0284331, 19.1174177, 19.2064092, 19.2954759, 19.3846673, 
        19.4740111, 19.563505, 19.6531313, 19.7428428, 19.8325747, 19.9223362, 20.0121912, 20.1021493, 20.1921695, 20.2822079, 20.3721968, 
        20.4620408, 20.551582, 20.6406968, 20.7294226, 20.8179344, 20.9063209, 20.9945894, 21.0826529, 21.1703971, 21.2577906, 21.3447998, 
        21.4313366, 21.5173477, 21.6028457, 21.6878123, 21.7722376, 21.8561388, 21.9394692, 22.0220622, 22.1038144, 22.1847138, 22.2647818, 
        22.3439753, 22.4222058, 22.4993417, 22.575224, 22.6496944, 22.7225271, 22.7934992, 22.8624036, 22.9290611, 22.9933236, 23.0550455, 
        23.1141048, 23.1704164, 23.2239275, 23.2745995, 23.3223554, 23.3671161, 23.408808, 23.44741, 23.482888, 23.5152325, 23.5445079, 
        23.5708406, 23.5943475, 23.615159, 23.6334368, 23.6493909, 23.6632086, 23.675082, 23.6851981, 23.693753, 23.7009191, 23.7068824, 
        23.7118098, 23.7158619, 23.7191778, 23.7218902, 23.7241109, 23.7259348, 23.7274342, 23.7286719, 23.7296984, 23.7305572, 23.7312815, 
        23.7318963, 23.7324238, 23.7328847, 23.7332973, 23.7336756, 23.7340288, 23.7343623, 23.7346809, 23.7349879, 23.7352866, 23.7355774, 
        23.7358612, 23.7361387, 23.7364119, 23.7366822, 23.7369512, 23.7372188, 23.7374856, 23.7377511, 23.7380162, 23.7382802, 23.7385437, 
        23.738806, 23.7390676, 23.7393279, 23.7395875, 23.7398457, 23.7401032, 23.7403592, 23.7406145, 23.7408683, 23.7411214, 23.7413729, 
        23.7416237, 23.7418728, 23.7421212, 23.742368, 23.742614, 23.7428583, 23.7431018, 23.7433435, 23.7435845, 23.7438237, 23.7440621, 
        23.7442987, 23.7445344, 23.7447684, 23.7450015, 23.7452328, 23.7454632, 23.7456918, 23.7459195, 23.7461453, 23.7463703, 23.7465934, 
        23.7468155, 23.7470359, 23.7472553, 23.7474728, 23.7476895, 23.7479042, 23.7481181, 23.7483301, 23.7485411, 23.7487503, 23.7489586, 
        23.7491649, 23.7493704, 23.749574, 23.7497767, 23.7499775, 23.7501774, 23.7503754, 23.7505725, 23.7507678, 23.7509621, 23.7511546, 
        23.7513463, 23.751536, 23.7517249, 23.751912, 23.7520982, 23.7522825, 23.752466, 23.7526477, 23.7528285, 23.7530076, 23.7531857, 
        23.7533621, 23.7535377, 23.7537115, 23.7538845, 23.7540557, 23.7542261, 23.7543948, 23.7545626, 23.7547288, 23.7548941, 23.7550578, 
        23.7552207, 23.7553819, 23.7555423, 23.7557011, 23.7558591, 23.7560154, 23.756171, 23.756325, 23.7564783, 23.75663, 23.7567809, 
        23.7569303, 23.7570789, 23.757226, 23.7573724, 23.7575172, 23.7576614, 23.7578041, 23.757946, 23.7580865, 23.7582263, 23.7583646, 
        23.7585023, 23.7586385, 23.7587741, 23.7589083, 23.7590418, 23.7591739, 23.7593054, 23.7594355, 23.759565, 23.7596931, 23.7598206, 
        23.7599468, 23.7600724, 23.7601967, 23.7603203, 23.7604428, 23.7605645, 23.7606851, 23.760805, 23.7609238, 23.7610419, 23.7611589, 
        23.7612752, 23.7613904, 23.7615049, 23.7616184, 23.7617312, 23.761843, 23.7619541, 23.7620642, 23.7621737, 23.7622821, 23.7623899, 
        23.7624968, 23.7626029, 23.7627082, 23.7628127, 23.7629164, 23.7630194, 23.7631216, 23.763223, 23.7633236, 23.7634236, 23.7635227, 
        23.7636211, 23.7637188, 23.7638157, 23.7639119, 23.7640074, 23.7641022, 23.7641963, 23.7642896, 23.7643823, 23.7644743, 23.7645655, 
        23.7646561, 23.7647461, 23.7648353, 23.7649239, 23.7650118, 23.765099, 23.7651857, 23.7652716, 23.7653569, 23.7654415, 23.7655256, 
        23.7656089, 23.7656917, 23.7657738, 23.7658554, 23.7659362, 23.7660165, 23.7660962, 23.7661753, 23.7662537, 23.7663316, 23.7664088, 
        23.7664855, 23.7665616, 23.7666371, 23.766712, 23.7667864, 23.7668601, 23.7669333, 23.7670059, 23.767078, 23.7671494, 23.7672204, 
        23.7672907, 23.7673605, 23.7674297, 23.7674985, 23.7675666, 23.7676342, 23.7677012, 23.7677678, 23.7678337, 23.7678992, 23.767964, 
        23.7680285, 23.7680923, 23.7681556, 23.7682183, 23.7682807, 23.7683423, 23.7684036, 23.7684642, 23.7685244, 23.768584, 23.7686432, 
        23.7687017, 23.7687599, 23.7688174, 23.7688746, 23.768931, 23.7689872, 23.7690426, 23.7690978, 23.7691522, 23.7692063, 23.7692598, 
        23.7693129, 23.7693653, 23.7694174, 23.7694688, 23.7695199, 23.7695704, 23.7696205, 23.7696699, 23.769719, 23.7697675, 23.7698156, 
        23.769863, 23.7699101, 23.7699566, 23.7700027, 23.7700482, 23.7700933, 23.7701379, 23.770182, 23.7702255, 23.7702687, 23.7703112, 
        23.7703534, 23.7703949, 23.7704361, 23.7704767, 23.7705168, 23.7705564, 23.7705956, 23.7706342, 23.7706724, 23.7707101, 23.7707473, 
        23.7707839, 23.7708202, 23.7708558, 23.770891, 23.7709257, 23.77096, 23.7709937, 23.7710269, 23.7710596, 23.7710919, 23.7711236, 
        23.7711548, 23.7711856, 23.7712158, 23.7712456, 23.7712748, 23.7713036, 23.7713318, 23.7713596, 23.7713868, 23.7714135, 23.7714398, 
        23.7714655, 23.7714907, 23.7715154, 23.7715396, 23.7715632, 23.7715863, 23.7716089, 23.7716309, 23.7716524, 23.7716733, 23.7716936, 
        23.7717134, 23.7717326, 23.7717511, 23.771769, 23.7717863, 23.7718029, 23.7718189, 23.7718342, 23.7718488, 23.7718626, 23.7718758, 
        23.7718882, 23.7718999, 23.7719108, 23.771921, 23.7719304, 23.7719391, 23.771947, 23.7719542, 23.7719607, 23.7719665, 23.7719717, 
        23.7719762, 23.7719801, 23.7719834, 23.7719863, 23.7719886, 23.7719905, 23.771992, 23.7719932, 23.7719941, 23.7719948, 23.7719952, 
        23.7719956, 23.7719957, 23.7719958, 23.7719959, 23.7719959, 23.7719959, 23.7719959};
protected:
    /** Slip wall boundary conditions (No penetration)
     *  * Given by Algorithm II of the following paper:
     *  * * Krivodonova, L., and Berger, M.,
     *      “High-order accurate implementation of solid wall boundary conditions in curved geometries,”
     *      Journal of Computational Physics, vol. 211, 2006, pp. 492–512.
     */
    void boundary_slip_wall (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Wall boundary condition
    void boundary_wall (
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Evaluate the manufactured solution boundary conditions.
    virtual void boundary_manufactured_solution (
        const dealii::Point<dim, real> &pos,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        const std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Pressure Outflow Boundary Condition (back pressure)
    /// Reference: Carlson 2011, sec. 2.4
    void boundary_pressure_outflow (
        const real total_inlet_pressure,
        const real back_pressure,
        const std::array<real,nstate> &soln_int,
        std::array<real,nstate> &soln_bc) const;

    /// Inflow boundary conditions (both subsonic and supersonic)
    /// Reference: Carlson 2011, sec. 2.2 & sec 2.9
    void boundary_inflow (
        const real total_inlet_pressure,
        const real total_inlet_temperature,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        std::array<real,nstate> &soln_bc) const;

    /// Inflow boundary conditions for turbulent boundary layer
    void boundary_inflow_turbulent_BL (
        const dealii::Point<dim, real> &pos,
        const real total_inlet_pressure,
        const real total_inlet_temperature,
        const dealii::Tensor<1,dim,real> &normal_int,
        const std::array<real,nstate> &soln_int,
        std::array<real,nstate> &soln_bc) const;

    /// Riemann-based farfield boundary conditions based on freestream values.
    /// Reference: ? (ask Doug)
    void boundary_riemann (
       const dealii::Tensor<1,dim,real> &normal_int,
       const std::array<real,nstate> &soln_int,
       std::array<real,nstate> &soln_bc) const;

    /// Simple farfield boundary conditions based on freestream values
    void boundary_farfield (
        std::array<real,nstate> &soln_bc) const;

    /// p0 extrapolation at the boundary
    void boundary_p0_extrapolation (
        const std::array<real,nstate> &soln_int,
        std::array<real,nstate> &soln_bc,
        std::array<dealii::Tensor<1,dim,real>,nstate> &soln_grad_bc) const;

    /// Custom boundary conditions for the left boundary of the astrophysical mach jet case where it is not hypersonic inflow.
    void boundary_custom (
        std::array<real,nstate> &soln_bc) const;

    /// Boundary conditions based on user-defined values
    void boundary_astrophysical_inflow (
        std::array<real,nstate> &soln_bc) const;

    /// Get manufactured solution value
    std::array<real,nstate> get_manufactured_solution_value(
        const dealii::Point<dim,real> &pos) const;

    /// Get manufactured solution gradient
    std::array<dealii::Tensor<1,dim,real>,nstate> get_manufactured_solution_gradient(
        const dealii::Point<dim,real> &pos) const;

    /** Entropy conserving split form flux of Kennedy and Gruber.
     *  Refer to Gassner's paper (2016) Eq. 3.10  */
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_kennedy_gruber (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Compute Ismail-Roe parameter vector from primitive solution
    std::array<real,nstate> compute_ismail_roe_parameter_vector_from_primitive(
        const std::array<real,nstate> &primitive_soln) const;

    /// Compute Ismail-Roe logarithmic mean
    real compute_ismail_roe_logarithmic_mean(const real val1, const real val2) const;

    /** Entropy conserving split form flux of Ismail & Roe.
     *  Refer to Gassner's paper (2016) Eq. 3.17  */
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_ismail_roe (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Chandrashekar entropy conserving flux.
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_chandrashekar (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;

    /// Ranocha pressure equilibrium preserving, entropy and energy conserving flux.
    std::array<dealii::Tensor<1,dim,real>,nstate> convective_numerical_split_flux_ranocha (
        const std::array<real,nstate> &conservative_soln1,
        const std::array<real,nstate> &conservative_soln2) const;
};

} // Physics namespace
} // PHiLiP namespace

#endif
