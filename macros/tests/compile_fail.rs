#[test]
fn test_extra_param_error() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/extra_param_in_docstring.rs");
}

#[test]
fn test_missing_param_error() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/missing_param_description.rs");
}

#[test]
fn test_duplicate_tool_error() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/duplicate_tool.rs");
}

#[test]
fn test_tool_parameter_requires_json_schema() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ui/unsupported_tool_parameter.rs");
}
