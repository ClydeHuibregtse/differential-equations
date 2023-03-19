use ndarray::{arr0, concatenate, Array, Array1, Axis, s};
use num::complex::Complex64;

pub fn forward_euler<D>(
    // f: fn(&Array<Complex64, D>, f64) -> Array<Complex64, D>,
    f: impl Fn(&Array<Complex64, D>, f64) -> Array<Complex64, D>,
    u0: Array<Complex64, D>,
    t0: f64,
    tf: f64,
    h: Complex64,
) -> (Vec<Array<Complex64, D>>, Vec<f64>)
where
    D: ndarray::Dimension,
{
    let mut us = Vec::new();
    let mut ts = Vec::new();
    let mut t = t0;
    us.push(u0);
    ts.push(t);
    let mut count = 0;
    while t < tf {
        let uf = &us[count];
        println!("{:?}", uf);
        us.push(uf + (h * &f(uf, t)));
        ts.push(t);
        t += h.re;
        count += 1;
    }
    (us, ts)
}

pub fn concatenate_vec_of_arrays<D>(us: Vec<Array<Complex64, D>>) -> Array<Complex64, D>
where
    D: ndarray::Dimension + ndarray::RemoveAxis,
{
    let mut arr_us = us[0].clone();
    for u in us[1..].iter() {
        arr_us = concatenate(Axis(0), &[arr_us.view(), u.view()])
            .unwrap()
            .clone();
    }
    arr_us
}

#[cfg(test)]
mod tests {
    use crate::plots::plot1d;
    use crate::solve::c_;
    use ndarray::{concatenate, stack, Array, Array1, ArrayBase, Axis, Dim, OwnedRepr, ViewRepr};
    use num::complex::Complex64;

    use super::{concatenate_vec_of_arrays, forward_euler};

    #[test]
    fn test_univariate_ode() {
        let u0 = Array1::from_vec(vec![c_(0.0, 0.0)]);
        let t0 = 0.0;
        let tf = 6.28 * 2.0;
        let h = c_(0.1, 0.0);
        let f = |x: &Array1<Complex64>, t: f64| Array1::from_vec(vec![c_(-t.sin(), 0.0)]);

        let (us, ts) = forward_euler(f, u0, t0, tf, h);
        plot1d(
            Array1::from_vec(ts),
            concatenate_vec_of_arrays(us),
            "test_plots/d_dx_cosx.png",
        )
        .unwrap();
    }
}
