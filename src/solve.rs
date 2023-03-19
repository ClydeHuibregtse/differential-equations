use std::{any::Any, clone};

use ndarray::{s, Array, Array1, Data};

use num::complex::{Complex, Complex32};

pub fn c_(alpha: f64, beta: f64) -> Complex<f64> {
    Complex::new(alpha, beta)
}

pub struct ODEProblem<T, D>
where
    D: ndarray::Dimension,
    T: Clone,
{
    pub f: fn(&Array<T, D>, f64) -> Array<T, D>,
    pub u0: Array<T, D>,
    pub tspan: (f64, f64),
}

pub struct ODESolution<T, D>
where
    D: ndarray::Dimension,
    T: Clone,
{
    u: Array<T, D>,
    t: Array1<f64>,
}

impl<T, D> ODEProblem<T, D>
where
    D: ndarray::Dimension,
    T: Clone,
{
    // pub fn new(f: fn(f64, f64) -> f64, u0: f64, tspan: (f64, f64)) -> ODEProblem<T, D> {
    //     ODEProblem { f: f, u0: u0, tspan: tspan }
    // }

    pub fn new(
        f: fn(&Array<T, D>, f64) -> Array<T, D>,
        u0: Array<T, D>,
        tspan: (f64, f64),
    ) -> ODEProblem<T, D> {
        ODEProblem {
            f: f,
            u0: u0,
            tspan: tspan,
        }
    }

    pub fn solve(&self) -> ODESolution<T, D>
    where
        D: ndarray::Dimension,
    {
        ODESolution {
            u: self.u0.clone(),
            t: Array::linspace(self.tspan.0, self.tspan.1, 100),
        }
    }
}

impl<T, D> ODESolution<T, D>
where
    D: ndarray::Dimension,
    T: Clone,
{
    pub fn u(&self) -> &Array<T, D> {
        &self.u
    }
}
