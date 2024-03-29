use std::iter::zip;

use ndarray::{Array1, Array2};
use num::complex::Complex64;
use plotters::prelude::*;

static COLORS:&'static [&RGBColor; 5]= &[
    &RED,
    &GREEN,
    &BLUE,
    &YELLOW,
    &CYAN,
];

pub fn plot1d(
    x: Array1<f64>,
    ys: Array2<Complex64>,
    fpath: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(fpath, (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;


    let xmax = *x.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() as f32;
    let xmin = *x.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap() as f32;

    let mut chart = ChartBuilder::on(&root)
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        // .build_cartesian_2d(xmin..xmax, ymin..ymax)?;
        .build_cartesian_2d(xmin..xmax, 0f32..30f32)?;
    chart.configure_mesh().draw()?;
    for (y, color) in ys.columns().into_iter().zip(COLORS.iter()) {
        // println!("{:?} {:?}", y, y.shape());
        // let ymax = y.iter().max_by(|x, y| x.re.partial_cmp(&y.re).unwrap()).unwrap().re as f32;
        // let ymin = y.iter().min_by(|x, y| x.re.partial_cmp(&y.re).unwrap()).unwrap().re as f32;
        // println!("{:?} {:?}", ymin, ymax);
        chart
            .draw_series(LineSeries::new(
                zip(x.map(|v| *v as f32), y.map(|v| v.re as f32)).into_iter(),
                color,
            ))?;
            // .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
    }
    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
