use pyo3::{exceptions::PySystemExit, prelude::*};

fn main() -> PyResult<()> {
	pyo3::prepare_freethreaded_python();

	Python::with_gil(|py| {
		let sys = py.import("sys")?;
		let args: Vec<_> = std::env::args().collect();
		let py_args = args.into_py(py);
		sys.setattr("argv", py_args)?;

		let bentoml_cli = PyModule::import(py, "bentoml_cli.cli")?;
		match bentoml_cli.getattr("cli")?.call0() {
			Ok(_) => {}
			Err(e) => {
				if e.is_instance_of::<PySystemExit>(py) {
					let exit_code = e
						.value(py)
						.getattr("code")
						.expect("Unable to get exit code from PySystemExit")
						.extract::<i32>()
						.expect("Unable to extract exit code from PySystemExit");
					std::process::exit(exit_code);
				}
				println!("Error: {}", e);
				match e.traceback(py) {
					Some(tb) => match tb.format() {
						Ok(s) => println!("{}", s),
						Err(e) => println!("Error formatting traceback: {}", e),
					},
					None => println!("No traceback available."),
				}

				// println!("This should never occur, please send this error log to the BentoML team!")
			}
		}
		Ok(())
	})
}
