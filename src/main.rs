use std::ffi::CString;

struct ONNXRunner {
    runner: *mut onnx_runner::ONNXRunner,
}

impl ONNXRunner {
    fn new(model: &str) -> Self {
        let c_str = CString::new(model).unwrap();
        let runner = unsafe { onnx_runner::onnx_runner_new(c_str.as_ptr()) };
        ONNXRunner { runner }
    }

    fn run(
        &mut self,
        input_shape: &[i64],
        output_shape: &[i64],
        input_names: &[&str],
        output_names: &[&str],
        input: &mut [f32],
        output: &mut [i64],
    ) {
        let c_input_shape = input_shape.as_ptr();
        let c_output_shape = output_shape.as_ptr();
        let c_input_names: Vec<CString> = input_names
            .iter()
            .map(|x| CString::new(*x).unwrap())
            .collect();
        let c_input_names_ptr: Vec<*const std::os::raw::c_char> =
            c_input_names.iter().map(|x| x.as_ptr()).collect();
        let c_output_names: Vec<CString> = output_names
            .iter()
            .map(|x| CString::new(*x).unwrap())
            .collect();
        let c_output_names_ptr: Vec<*const std::os::raw::c_char> =
            c_output_names.iter().map(|x| x.as_ptr()).collect();
        let c_input = input.as_mut_ptr();
        let c_output = output.as_mut_ptr();
        unsafe {
            onnx_runner::onnx_runner_run(
                self.runner,
                c_input_shape,
                input_shape.len() as u64,
                c_output_shape,
                output_shape.len() as u64,
                c_input_names_ptr.as_ptr(),
                input_names.len() as u64,
                c_output_names_ptr.as_ptr(),
                output_names.len() as u64,
                c_input,
                input.len() as u64,
                c_output,
                output.len() as u64,
            );
        }
    }
}

impl Drop for ONNXRunner {
    fn drop(&mut self) {
        unsafe { onnx_runner::onnx_runner_free(self.runner) };
    }
}

fn main() {
    let mut runner = ONNXRunner::new("iris.onnx");

    let mut input = [
        6.1, 2.8, 4.7, 1.2, 5.7, 3.8, 1.7, 0.3, 7.7, 2.6, 6.9, 2.3, 6., 2.9, 4.5, 1.5, 6.8, 2.8,
        4.8, 1.4, 5.4, 3.4, 1.5, 0.4, 5.6, 2.9, 3.6, 1.3, 6.9, 3.1, 5.1, 2.3, 6.2, 2.2, 4.5, 1.5,
        5.8, 2.7, 3.9, 1.2,
    ];
    let mut output = [0; 10];
    runner.run(
        &[10, 4],
        &[10],
        &["inputs"],
        &["label"],
        &mut input,
        &mut output,
    );

    println!("{:?}", output);
}
