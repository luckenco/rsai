/// Convert a snake_case name to PascalCase.
pub fn to_pascal_case(name: &str) -> String {
    name.split('_')
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect()
}
