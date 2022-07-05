
use rayon::iter::{ParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, IndexedParallelIterator, IntoParallelRefIterator};

const ITER: usize = 10;

struct Map<const W: usize, const H: usize>(Vec<f32>);

impl<const W: usize, const H: usize> Map<W, H> {
	fn new() -> Self {
		Self(vec![0.0; W * H])
	}
	
	fn from_raw(raw: Vec<f32>) -> Self {
		Self(raw)
	}
	
	fn get(&self, x: usize, y: usize) -> f32 {
		*self.0.get(x + y * W).unwrap_or(&0.0)
	}
	
	fn set(&mut self, x: usize, y: usize, val: f32) {
		if let Some(cell) = self.0.get_mut(x + y * W) { *cell = val };
	}
}

fn xi<const W: usize>(i: usize) -> (usize, usize) {
	(i % W, i / W)
}

fn getns<const W: usize, const H: usize>(arrx: &Map<W, H>, arry: &Map<W, H>, i: usize, j: usize) -> (f32, f32, f32, f32) {
	(
		arrx.get(i + 1, j),
		if i == 0 { 0.0 } else { arrx.get(i - 1, j) },
		arry.get(i, j + 1),
		if j == 0 { 0.0 } else { arry.get(i, j - 1) },
	)
}

fn diffuse<const W: usize, const H: usize>(b: u8, x: &mut Map<W, H>, x0: &Map<W, H>, diff: f32, dt: f32) {
	let a = dt * diff * (W - 2) as f32 * (H - 2) as f32;
	lin_solve(b, x, x0, a, 1.0 + 6.0 * a);
}

fn lin_solve<const W: usize, const H: usize>(b: u8, x: &mut Map<W, H>, x0: &Map<W, H>, a: f32, c: f32) {
	for _ in 0..ITER {
		*x = Map::<W, H>::from_raw((0..x.0.len()).into_par_iter().map(|i| {
			let (i, j) = xi::<W>(i);
			let ns = getns(x, x, i, j);
			let a1 = a * (ns.0 + ns.1 + ns.2 + ns.3);
			(x0.get(i, j) + a1) / c
		}).collect());
		
		set_bnd(b, x);
	}
}

fn project<const W: usize, const H: usize>(vx: &mut Map<W, H>, vy: &mut Map<W, H>, p: &mut Map<W, H>, div: &mut Map<W, H>) {
	div.0.par_iter_mut().zip(p.0.par_iter_mut()).enumerate().for_each(|(i, (div_cell, p_cell))| {
		let (i, j) = xi::<W>(i);
		let ns = getns(vx, vy, i, j);
		*div_cell = (ns.0 - ns.1 + ns.2 - ns.3) * -0.5 / H as f32;
		*p_cell = 0.0;
	});
	
	set_bnd(0, div);
	set_bnd(0, p);
	lin_solve(0, p, div, 1.0, 6.0);
	
	let (nvx, nvy) = vx.0.par_iter().zip(vy.0.par_iter()).enumerate().map(|(i, (vx_cell, vy_cell))| {
		let (i, j) = xi::<W>(i);
		let ns = getns(p, p, i, j);
		(vx_cell - 0.5 * (ns.0 - ns.1) * W as f32, vy_cell - 0.5 * (ns.2 - ns.3) * H as f32)
	}).unzip();
	
	*vx = Map::from_raw(nvx);
	*vy = Map::from_raw(nvy);
	
	set_bnd(1, vx);
	set_bnd(2, vy);
}

fn advect<const W: usize, const H: usize>(b: u8, d: &mut Map<W, H>, d0: &Map<W, H>, vx: &Map<W, H>, vy: &Map<W, H>, dt: f32) {
	let dtx = dt * (W - 2) as f32;
	let dty = dt * (H - 2) as f32;
	
	d.0.par_iter_mut().enumerate()
	.map(|(i, cell)| (xi::<W>(i), cell))
	.filter(|&((i, j), _)| i != 1 && j != 1 && i != W - 1 && j != W - 1)
	.for_each(|((i, j), cell)| {
		let x = (i as f32 - dtx * vx.get(i, j)).clamp(0.5, W as f32 + 0.5);
		let y = (j as f32 - dty * vy.get(i, j)).clamp(0.5, H as f32 + 0.5);
		
		let i0 = x.floor() as usize;
		let j0 = y.floor() as usize;
		
		let s1 = x - i0 as f32;
		let t1 = y - j0 as f32;
		
		*cell =
			(1.0 - s1) * ((1.0 - t1) * d0.get(i0    , j0) + t1 * d0.get(i0    , j0 + 1)) +
			s1         * ((1.0 - t1) * d0.get(i0 + 1, j0) + t1 * d0.get(i0 + 1, j0 + 1));
	});
	
	set_bnd(b, d);
}

fn set_bnd<const W: usize, const H: usize>(b: u8, x: &mut Map<W, H>) {
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
	diff: f32,
	visc: f32,
	s: Map<W, H>,
	density: Map<W, H>,
	vx: Map<W, H>,
	vy: Map<W, H>,
	vx0: Map<W, H>,
	vy0: Map<W, H>,
}

impl<const W: usize, const H: usize> Fluid<W, H> {
	pub fn new(diff: f32, visc: f32) -> Self {
		Self {
			diff,
			visc,
			s: Map::new(),
			density: Map::new(),
			vx: Map::new(),
			vy: Map::new(),
			vx0: Map::new(),
			vy0: Map::new(),
		}
	}
	
	pub fn step(&mut self, dt: f32) {
		diffuse(1, &mut self.vx0, &self.vx, self.visc, dt);
		diffuse(2, &mut self.vy0, &self.vy, self.visc, dt);
		
		project(&mut self.vx0, &mut self.vy0, &mut self.vx, &mut self.vy);
		
		advect(1, &mut self.vx, &self.vx0, &self.vx0, &self.vy0, dt);
		advect(2, &mut self.vy, &self.vy0, &self.vx0, &self.vy0, dt);
		
		project(&mut self.vx, &mut self.vy, &mut self.vx0, &mut self.vy0);
		diffuse(0, &mut self.s, &self.density, self.diff, dt);
		advect(0, &mut self.density, &mut self.s, &self.vx, &self.vy, dt);
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

	#[allow(dead_code)]
	pub fn for_every_cell<F : Fn(&mut f32, (&mut f32, &mut f32)) + Sync>(&mut self, f: F) {
		let d = &mut self.density.0;
		let vx = &mut self.vx.0;
		let vy = &mut self.vy.0;
		d.par_iter_mut()
			.zip(vx.par_iter_mut())
			.zip(vy.par_iter_mut())
			.for_each(|((d_cell, vx_cell), vy_cell)| f(d_cell, (vx_cell, vy_cell)));
	}
}
