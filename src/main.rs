
mod fluid;

use fluid::Fluid;
use nannou::{prelude::*, image::{ImageBuffer, Rgb, DynamicImage}, wgpu::Texture};

const WIDTH: usize = 80;
const HEIGHT: usize = 80;

const DT: f32 = 0.05;
const DIFF: f32 = 1e-4;
const VISC: f32 = 1e-8;

const FM: f32 = 0.2;

struct Model {
	time: f32,
	paused: bool,
	pressing: bool,
	prev_mouse: Point2,
	fluid: Fluid<WIDTH, HEIGHT>,
}

fn setup(_app: &App) -> Model {
	Model {
		time: 0.0,
		paused: false,
		pressing: false,
		prev_mouse: pt2(0.0, 0.0),
		fluid: Fluid::new(DT, DIFF, VISC),
	}
}

fn update(app: &App, model: &mut Model, event: Event) {
	match event {
		Event::WindowEvent { simple: Some(event), .. } => {
			match event {
				KeyPressed(Key::Space) => model.paused = !model.paused,
				KeyPressed(Key::R) => *model = setup(app),
				MousePressed(_) => model.pressing = true,
				MouseReleased(_) => model.pressing = false,
				MouseMoved(p) => {
					let prev_mouse = model.prev_mouse;
					model.prev_mouse = p;

					if model.pressing && !model.paused {
						let mdx = p.x - prev_mouse.x;
						let mdy = p.y - prev_mouse.y;

						let win = app.window_rect();

						let cx = map_range(p.x, win.left(), win.right(), 0, WIDTH) as usize;
						let cy = map_range(p.y, win.top(), win.bottom(), 0, HEIGHT) as usize;

						const R: isize = 3;

						for x in -R..=R {
							let y_max = ((R as f32).pow(2.0) - (x as f32).pow(2.0)).sqrt() as usize;
							for y in (cy - y_max)..=(cy + y_max) {
								model.fluid.add_density((cx as isize + x) as usize, y, 255.0);
							}
						}

						model.fluid.add_velocity(cx as usize, cy as usize, mdx * FM, -mdy * FM);
					}
				}
				_ => {}
			}
		}
		Event::Update(update) if !model.paused => {
			model.time += update.since_last.as_secs_f32();
			model.fluid.step();
		}
		_ => {}
	}
}

fn view(app: &App, model: &Model, frame: Frame) {
	let draw = app.draw();

	let img = ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
		let d = model.fluid.get_density_at(x as usize, y as usize) as u8;
		Rgb([d; 3])
	});

	let tex = Texture::from_image(app, &DynamicImage::ImageRgb8(img));

	draw.texture(&tex).w_h(512.0, 512.0);

	draw.to_frame(app, &frame).unwrap();
}

fn main() {
	nannou::app(setup)
	.event(update)
	.simple_window(view)
	.size(512, 512)
	.run();
}
