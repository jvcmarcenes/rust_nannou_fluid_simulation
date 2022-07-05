use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator, IntoParallelIterator};


const ITER: usize = 16;

struct Map<T: Copy, const W: usize, const H: usize>(Vec<T>);

impl<T: Copy, const W: usize, const H: usize> Map<T, W, H> {
	fn new(def: T) -> Self {
		Self(vec![def; W * H])
	}

	fn from_raw(raw: Vec<T>) -> Self {
		Self(raw)
	}

	fn get(&self, x: usize, y: usize) -> T {
		self.0[x + y * W]
	}

	fn set(&mut self, x: usize, y: usize, val: T) {
		self.0[x + y * W] = val;
	}
}

fn xi<const W: usize>(i: usize) -> (usize, usize) {
	(i % W, i / W)
}

fn getns<T : Copy, const W: usize, const H: usize>(arrx: &Map<T, W, H>, arry: &Map<T, W, H>, i: usize, j: usize) -> (T, T, T, T) {
	(arrx.get(i + 1, j), arrx.get(i - 1, j), arry.get(i, j + 1), arry.get(i, j - 1))
}

fn diffuse<const W: usize, const H: usize>(b: u8, x: &mut Map<f32, W, H>, x0: &Map<f32, W, H>, diff: f32, dt: f32) {
	let a = dt * diff * (W - 2) as f32 * (H - 2) as f32;
	lin_solve(b, x, x0, a, 1.0 + 6.0 * a);
}

fn lin_solve<const W: usize, const H: usize>(b: u8, x: &mut Map<f32, W, H>, x0: &Map<f32, W, H>, a: f32, c: f32) {
	for _ in 0..ITER {
		for j in 1..(H - 1) {
			for i in 1..(W - 1) {
				let ns = getns(x, x, i, j);
				let a1 = a * (ns.0 + ns.1 + ns.2 + ns.3);
				x.set(i, j, (x0.get(i, j) + a1) / c);
			}
		}
		// *x = Map::<f32, W, H>::from_raw((0..x.0.len()).into_par_iter().map(|i| {
		// 	let (i, j) = xi::<W>(i);
		// 	let ns = getns(x, x, i, j);
		// 	let a1 = a * (ns.0 + ns.1 + ns.2 + ns.3);
		// 	(x0.get(i, j) + a1) / c
		// }).collect());
		
		set_bnd(b, x);
	}
}

fn project<const W: usize, const H: usize>(vx: &mut Map<f32, W, H>, vy: &mut Map<f32, W, H>, p: &mut Map<f32, W, H>, div: &mut Map<f32, W, H>) {
	for j in 1..(H - 1) {
		for i in 1..(W - 1) {
			let ns = getns(vx, vy, i, j);
			div.set(i, j, (ns.0 - ns.1 + ns.2 - ns.3) * -0.5 / H as f32);
			p.set(i, j, 0.0);
		}
	}

	set_bnd(0, div);
	set_bnd(0, p);
	lin_solve(0, p, div, 1.0, 6.0);

	for j in 1..(H - 1) {
		for i in 1..(W - 1) {
			let ns = getns(p, p, i, j);
			vx.set(i, j, vx.get(i, j) - 0.5 * (ns.0 - ns.1) * W as f32);
			vy.set(i, j, vy.get(i, j) - 0.5 * (ns.2 - ns.3) * H as f32);
		}
	}

	set_bnd(1, vx);
	set_bnd(2, vy);
}

fn advect<const W: usize, const H: usize>(b: u8, d: &mut Map<f32, W, H>, d0: &Map<f32, W, H>, vx: &Map<f32, W, H>, vy: &Map<f32, W, H>, dt: f32) {
	let dtx = dt * (W - 2) as f32;
	let dty = dt * (H - 2) as f32;

	for j in 1..(H - 1) {
		for i in 1..(W - 1) {
			let tmpx = dtx * vx.get(i, j);
			let tmpy = dty * vy.get(i, j);

			let x = (i as f32 - tmpx).clamp(0.5, W as f32 + 0.5);
			let i0 = x.floor() as usize;
			let i1 = i0 + 1;

			let y = (j as f32 - tmpy).clamp(0.5, H as f32 + 0.5);
			let j0 = y.floor() as usize;
			let j1 = j0 + 1;

			let s1 = x - i0 as f32;
			let s0 = 1.0 - s1;
			let t1 = y - j0 as f32;
			let t0 = 1.0 - t1;

			let i0i = (i0 as usize).clamp(0, W - 1);
			let i1i = (i1 as usize).clamp(0, W - 1);
			let j0i = (j0 as usize).clamp(0, H - 1);
			let j1i = (j1 as usize).clamp(0, H - 1);

			d.set(i, j,
				s0 * (t0 * d0.get(i0i, j0i) + t1 * d0.get(i0i, j1i)) +
				s1 * (t0 * d0.get(i1i, j0i) + t1 * d0.get(i1i, j1i))
			);
		}
	}

	set_bnd(b, d);
}

fn set_bnd<const W: usize, const H: usize>(b: u8, x: &mut Map<f32, W, H>) {
	for i in 1..(W - 1) {
		x.set(i, 0, if b == 2 { -1.0 } else { 1.0 } * x.get(i, 1));
		x.set(i, H - 1, if b == 2 { -1.0 } else { 1.0 } * x.get(i, H - 2));
	}

	for j in 1..(H - 1) {
		x.set(0, j, if b == 1 { -1.0 } else { 1.0 } * x.get(1, j));
		x.set(W - 1, j, if b == 1 { -1.0 } else { 1.0 } * x.get(W - 2, j));
	}

	x.set(0, 0, 0.5 * (x.get(1, 0) + x.get(0, 1)));
	x.set(0, H - 1, 0.5 * (x.get(1, H - 1) + x.get(0, H - 2)));
	x.set(W - 1, 0, 0.5 * (x.get(W - 2, 0) + x.get(W - 1, 1)));
	x.set(W - 1, H - 1, 0.5 * (x.get(W - 2, H - 1) + x.get(W - 1, H - 2)));
}

pub struct Fluid<const W: usize, const H: usize> {
	dt: f32,
	diff: f32,
	visc: f32,
	s: Map<f32, W, H>,
	density: Map<f32, W, H>,
	vx: Map<f32, W, H>,
	vy: Map<f32, W, H>,
	vx0: Map<f32, W, H>,
	vy0: Map<f32, W, H>,
}

impl<const W: usize, const H: usize> Fluid<W, H> {
	pub fn new(dt: f32, diff: f32, visc: f32) -> Self {
		Self {
			dt,
			diff,
			visc,
			s: Map::new(0.0),
			density: Map::new(0.0),
			vx: Map::new(0.0),
			vy: Map::new(0.0),
			vx0: Map::new(0.0),
			vy0: Map::new(0.0),
		}
	}

	pub fn step(&mut self) {
		diffuse(1, &mut self.vx0, &self.vx, self.visc, self.dt);
		diffuse(2, &mut self.vy0, &self.vy, self.visc, self.dt);

		project(&mut self.vx0, &mut self.vy0, &mut self.vx, &mut self.vy);

		advect(1, &mut self.vx, &self.vx0, &self.vx0, &self.vy0, self.dt);
		advect(2, &mut self.vy, &self.vy0, &self.vx0, &self.vy0, self.dt);

		project(&mut self.vx, &mut self.vy, &mut self.vx0, &mut self.vy0);
		diffuse(0, &mut self.s, &self.density, self.diff, self.dt);
		advect(0, &mut self.density, &mut self.s, &self.vx, &self.vy, self.dt);
	}

	pub fn add_density(&mut self, x: usize, y: usize, amount: f32) {
		if x >= W || y >= H { return; }
		self.density.set(x, y, (self.density.get(x, y) + amount).clamp(0.0, 255.0));
	}
	
	pub fn add_velocity(&mut self, x: usize, y: usize, ax: f32, ay: f32) {
		if x >= W || y >= H { return; }
		self.vx.set(x, y, self.vx.get(x, y) + ax);
		self.vy.set(x, y, self.vy.get(x, y) + ay);
	}

	#[allow(dead_code)]
	pub fn get_density_at(&self, x: usize, y: usize) -> f32 {
		self.density.get(x, y)
	}

	#[allow(dead_code)]
	pub fn get_vel_at(&self, x: usize, y: usize) -> (f32, f32) {
		(self.vx.get(x, y), self.vy.get(x, y))
	}
}
