
mod fluid;

use fluid::Fluid;
use nannou::{prelude::*, image::{ImageBuffer, Rgb, DynamicImage}, wgpu::Texture};

const WIDTH: usize = 128;
const HEIGHT: usize = 128;

const DIFF: f32 = 1e-6;
const VISC: f32 = 1e-7;

const FM: f32 = 0.18;

const BRUSH_RAD: isize = 8;
const BRUSH_STRENGTH: f32 = 255.0;

const SPEED: f32 = 1.0;

enum RenderMode { Density, Velocity }

struct Model {
	paused: bool,
	left_pressed: bool,
	right_pressed: bool,
	render_mode: RenderMode,
	prev_mouse: Point2,
	fluid: Fluid<WIDTH, HEIGHT>,
}

fn setup(_app: &App) -> Model {
	Model {
		paused: false,
		left_pressed: false,
		right_pressed: false,
		render_mode: RenderMode::Density,
		prev_mouse: pt2(0.0, 0.0),
		fluid: Fluid::new(DIFF, VISC),
	}
}

fn update(app: &App, model: &mut Model, event: Event) {
	match event {
		Event::WindowEvent { simple: Some(event), .. } => {
			match event {
				KeyPressed(Key::Space) => model.paused = !model.paused,
				KeyPressed(Key::R) => *model = setup(app),
				KeyPressed(Key::D) => model.render_mode = RenderMode::Density,
				KeyPressed(Key::V) => model.render_mode = RenderMode::Velocity,
				MousePressed(MouseButton::Left) => model.left_pressed = true,
				MouseReleased(MouseButton::Left) => model.left_pressed = false,
				MousePressed(MouseButton::Right) => model.right_pressed = true,
				MouseReleased(MouseButton::Right) => model.right_pressed = false,
				MouseMoved(p) => {
					let prev_mouse = model.prev_mouse;
					model.prev_mouse = p;

					if model.right_pressed || model.left_pressed {
						let mdx = p.x - prev_mouse.x;
						let mdy = p.y - prev_mouse.y;

						let win = app.window_rect();

						let cx = map_range(model.prev_mouse.x.clamp(win.left(), win.right()), win.left(), win.right(), 0, WIDTH) as usize;
						let cy = map_range(model.prev_mouse.y.clamp(win.bottom(), win.top()), win.top(), win.bottom(), 0, HEIGHT) as usize;

						model.fluid.add_velocity(cx as usize, cy as usize, mdx * FM, -mdy * FM);
					}
				}
				_ => {}
			}
		}
		Event::Update(update) => {

			if model.left_pressed {
				let win = app.window_rect();
	
				let cx = map_range(model.prev_mouse.x, win.left(), win.right(), 0, WIDTH) as usize;
				let cy = map_range(model.prev_mouse.y, win.top(), win.bottom(), 0, HEIGHT) as usize;
	
				for x in -BRUSH_RAD..=BRUSH_RAD {
					let y_max = ((BRUSH_RAD as f32).pow(2.0) - (x as f32).pow(2.0)).sqrt() as isize;
					for y in -y_max..=y_max {
						// let d = 1.0 - (((x as f32).pow(2.0) + (y as f32).pow(2.0)).sqrt() / 3.0).pow(8.0);
						let x = (cx as isize + x).clamp(0, isize::MAX) as usize;
						let y = (cy as isize + y).clamp(0, isize::MAX) as usize;
						model.fluid.add_density(x, y, BRUSH_STRENGTH);
					}
				}
			}

			if !model.paused {
				model.fluid.step(SPEED * update.since_last.as_secs_f32());
			}

		}
		_ => {}
	}
}

fn view(app: &App, model: &Model, frame: Frame) {
	let draw = app.draw();

	draw.background().color(WHITE);

	let img = match model.render_mode {
		RenderMode::Density => ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
			let d = model.fluid.get_density_at(x as usize, y as usize) as u8;
			Rgb([255 - d; 3])
		}),
		RenderMode::Velocity => ImageBuffer::from_fn(WIDTH as u32, HEIGHT as u32, |x, y| {
			let (vx, vy) = model.fluid.get_vel_at(x as usize, y as usize);
			let vx = map_range(vx.clamp(-0.1, 0.1), -0.1, 0.1, 0, 255);
			let vy = map_range(vy.clamp(-0.1, 0.1), -0.1, 0.1, 0, 255);
			Rgb([vx as u8, vy as u8, 255])
		}),
	};

	let tex = Texture::from_image(app, &DynamicImage::ImageRgb8(img));

	draw.texture(&tex).w_h(512.0, 512.0);

	draw.to_frame(app, &frame).unwrap();
}

fn main() {
	nannou::app(setup)
	.event(update)
	.simple_window(view)
	.size(496, 496)
	.run();
}
