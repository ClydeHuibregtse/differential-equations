mod algorithms;
mod plots;
mod solve;

use crate::solve::{ODEProblem, ODESolution};
use ndarray::{arr0, Array, Array0, Array1};

fn main() {
    println!("Hello, world!");
    let a = Array::<f64, _>::zeros(10);
    println!("{}", 2.0 * (&a + 1.0));
}

#[cfg(test)]
mod tests {
    use crate::solve::c_;
    use crate::solve::ODEProblem;
    use ndarray::{arr0, Array, Array0, Array1};
    use num::complex::Complex64;

    #[test]
    fn problem_init() {
        fn du(u: &Array1<f64>, t: f64) -> Array1<f64> {
            -u
        }
        let prob = ODEProblem {
            f: du,
            u0: Array::zeros(100),
            tspan: (0f64, 1f64),
        };

        fn du_scalar(u: &Array0<f64>, t: f64) -> Array0<f64> {
            -u
        }
        let prob2 = ODEProblem {
            f: du_scalar,
            u0: arr0(1.0),
            tspan: (0f64, 1f64),
        };

        fn du_complex(u: &Array0<Complex64>, t: f64) -> Array0<Complex64> {
            -u
        }
        let prob2 = ODEProblem {
            f: du_complex,
            u0: arr0(c_(1.0, 1.0)),
            tspan: (0f64, 1f64),
        };
    }
}
