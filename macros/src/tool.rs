use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, FnArg, ItemFn, Pat, Result, Type};

/// Information about a context parameter (marked with #[context])
struct ContextParam {
    name: String,
    /// The inner type of the reference (e.g., for `&DatabasePool`, this is `DatabasePool`)
    inner_ty: Type,
}

pub fn tool_impl(attr: TokenStream, item: TokenStream) -> Result<TokenStream> {
    let _ = attr; // Currently unused, could be used for tool configuration

    let input = syn::parse2::<ItemFn>(item)?;
    let fn_name = &input.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Extract function description and parameter descriptions from doc comments
    let (description, param_descriptions) = extract_doc_comment_and_params(&input.attrs);

    // Parse function parameters, separating context params from regular params
    let (context_param, params) = parse_parameters(&input.sig.inputs, &param_descriptions)?;

    // Validate that all docstring parameters exist as actual parameters (skip context params)
    validate_parameter_descriptions(&params, &param_descriptions, &input.sig)?;

    // Generate the wrapper struct name
    let wrapper_name = quote::format_ident!(
        "{}Tool",
        crate::common::to_pascal_case(&fn_name.to_string())
    );
    let params_name = quote::format_ident!(
        "__Rsai{}Parameters",
        crate::common::to_pascal_case(&fn_name.to_string())
    );
    let params_struct = generate_parameter_struct(&params_name, &params);
    let schema = quote! {
        rsai::__private::serde_json::to_value(
            rsai::__private::schemars::schema_for!(#params_name)
        ).expect("generated tool schema must serialize")
    };

    // Check if function is async
    let is_async = input.sig.asyncness.is_some();

    // Generate the execution code
    let execute_impl = generate_execute_impl(
        fn_name,
        &context_param,
        &params,
        &params_name,
        is_async,
        ContextAccess::Borrowed,
    )?;
    let execute_owned_impl = if is_async {
        quote! {}
    } else {
        let owned_execute_impl = generate_execute_impl(
            fn_name,
            &context_param,
            &params,
            &params_name,
            is_async,
            ContextAccess::OwnedArc,
        )?;
        let unused_owned_ctx = if context_param.is_none() {
            quote! {
                let _ = __ctx;
            }
        } else {
            quote! {}
        };

        quote! {
            fn execute_owned(
                self: ::std::sync::Arc<Self>,
                __ctx: ::std::sync::Arc<__Ctx>,
                params: rsai::__private::serde_json::Value,
            ) -> rsai::BoxFuture<'static, Result<rsai::__private::serde_json::Value, rsai::LlmError>>
            where
                Self: 'static,
                __Ctx: Send + Sync + 'static,
            {
                use rsai::{BoxFuture, LlmError};
                let _ = self;
                #unused_owned_ctx
                Box::pin(async move {
                    rsai::__private::spawn_blocking_tool(move || {
                        #owned_execute_impl
                    }).await
                })
            }
        }
    };

    // Generate inherent impl with schema() method.
    // This allows calling .schema() without type annotations since Rust prefers
    // inherent methods over trait methods during method resolution.
    let inherent_impl = quote! {
        impl #wrapper_name {
            pub fn schema(&self) -> rsai::Tool {
                use rsai::Tool;
                Tool {
                    name: #fn_name_str.to_string(),
                    description: #description,
                    parameters: #schema,
                    strict: Some(true),
                }
            }
        }
    };

    // Generate trait implementation based on whether there's a context param
    let trait_impl = if let Some(ctx_param) = &context_param {
        // Tool with context: impl<Ctx> ToolFunction<Ctx> for Tool where Ctx: AsRef<ContextType>
        let ctx_inner_ty = &ctx_param.inner_ty;
        quote! {
            impl<__Ctx> rsai::ToolFunction<__Ctx> for #wrapper_name
            where
                __Ctx: AsRef<#ctx_inner_ty> + Send + Sync,
            {
                fn schema(&self) -> rsai::Tool {
                    #wrapper_name::schema(self)
                }

                fn execute<'a>(&'a self, __ctx: &'a __Ctx, params: rsai::__private::serde_json::Value) -> rsai::BoxFuture<'a, Result<rsai::__private::serde_json::Value, rsai::LlmError>> {
                    use rsai::{BoxFuture, LlmError};
                    Box::pin(async move {
                        #execute_impl
                    })
                }

                #execute_owned_impl
            }
        }
    } else {
        // Tool without context: impl<Ctx> ToolFunction<Ctx> for Tool (context is ignored)
        quote! {
            impl<__Ctx: Send + Sync> rsai::ToolFunction<__Ctx> for #wrapper_name {
                fn schema(&self) -> rsai::Tool {
                    #wrapper_name::schema(self)
                }

                fn execute<'a>(&'a self, __ctx: &'a __Ctx, params: rsai::__private::serde_json::Value) -> rsai::BoxFuture<'a, Result<rsai::__private::serde_json::Value, rsai::LlmError>> {
                    use rsai::{BoxFuture, LlmError};
                    let _ = __ctx; // Unused for context-free tools
                    Box::pin(async move {
                        #execute_impl
                    })
                }

                #execute_owned_impl
            }
        }
    };

    // Generate the complete implementation
    let expanded = quote! {
        #input

        #params_struct

        #[derive(Clone)]
        pub struct #wrapper_name;

        #inherent_impl

        #trait_impl
    };

    Ok(expanded)
}

fn extract_doc_comment_and_params(
    attrs: &[Attribute],
) -> (TokenStream, std::collections::HashMap<String, String>) {
    let doc_strings: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if attr.path().is_ident("doc")
                && let syn::Meta::NameValue(meta) = &attr.meta
                && let syn::Expr::Lit(expr_lit) = &meta.value
                && let syn::Lit::Str(lit_str) = &expr_lit.lit
            {
                return Some(lit_str.value().trim().to_string());
            }
            None
        })
        .collect();

    let mut description_lines = Vec::new();
    let mut param_descriptions = std::collections::HashMap::new();

    for line in doc_strings {
        // Check if this line describes a parameter (format: "param_name: description")
        if let Some(colon_pos) = line.find(':') {
            let param_name = line[..colon_pos].trim();
            let param_desc = line[colon_pos + 1..].trim();

            // Only treat it as a parameter description if the param name looks like an identifier
            if param_name.chars().all(|c| c.is_alphanumeric() || c == '_') && !param_name.is_empty()
            {
                param_descriptions.insert(param_name.to_string(), param_desc.to_string());
                continue;
            }
        }

        // Otherwise, it's part of the function description
        description_lines.push(line);
    }

    let description = if description_lines.is_empty() {
        quote! { None }
    } else {
        let description = description_lines.join(" ");
        quote! { Some(#description.to_string()) }
    };

    (description, param_descriptions)
}

struct Parameter {
    name: String,
    ty: Type,
    description: Option<String>,
    required: bool,
}

/// Check if a type is `Ctx<T>` and extract the inner type.
/// Returns Some((inner_type, ref_inner_type)) if it's a Ctx wrapper, None otherwise.
/// inner_type is the full type inside Ctx (e.g., `&DatabasePool`)
/// ref_inner_type is the type without the reference (e.g., `DatabasePool`)
fn extract_ctx_type(ty: &Type) -> Option<(Type, Type)> {
    if let Type::Path(type_path) = ty
        && let segments = &type_path.path.segments
        && segments.len() == 1
        && segments[0].ident == "Ctx"
        && let syn::PathArguments::AngleBracketed(args) = &segments[0].arguments
        && let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first()
        && let Type::Reference(type_ref) = inner_ty
    {
        // inner_ty is the type inside Ctx<>, which should be a reference like &DatabasePool
        // ref_inner is the type without the reference (e.g., DatabasePool)
        return Some((inner_ty.clone(), (*type_ref.elem).clone()));
    }
    None
}

fn parse_parameters(
    inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
    param_descriptions: &std::collections::HashMap<String, String>,
) -> Result<(Option<ContextParam>, Vec<Parameter>)> {
    let mut params = Vec::new();
    let mut context_param: Option<ContextParam> = None;

    for arg in inputs {
        match arg {
            FnArg::Receiver(_) => {
                return Err(syn::Error::new_spanned(
                    arg,
                    "tool functions cannot have self parameter",
                ));
            }
            FnArg::Typed(pat_type) => {
                let name = match &*pat_type.pat {
                    Pat::Ident(pat_ident) => pat_ident.ident.to_string(),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &pat_type.pat,
                            "only simple identifiers are supported for parameters",
                        ));
                    }
                };

                // Check if this is a context parameter (Ctx<&T> type)
                if let Some((_full_inner, ref_inner)) = extract_ctx_type(&pat_type.ty) {
                    if context_param.is_some() {
                        return Err(syn::Error::new_spanned(
                            pat_type,
                            "only one Ctx<&T> parameter is allowed per tool function",
                        ));
                    }
                    context_param = Some(ContextParam {
                        name,
                        inner_ty: ref_inner,
                    });
                    continue; // Don't add to regular params
                }

                // Get parameter description from docstring parsing
                let description = param_descriptions.get(&name).cloned();

                // Check if type is Option<T>
                let (ty, required) = match &*pat_type.ty {
                    Type::Path(type_path) => type_path
                        .path
                        .segments
                        .last()
                        .filter(|segment| segment.ident == "Option")
                        .and_then(|segment| match &segment.arguments {
                            syn::PathArguments::AngleBracketed(args) => args.args.first(),
                            _ => None,
                        })
                        .and_then(|argument| match argument {
                            syn::GenericArgument::Type(inner_ty) => Some(inner_ty.clone()),
                            _ => None,
                        })
                        .map_or(((*pat_type.ty).clone(), true), |inner_ty| (inner_ty, false)),
                    _ => ((*pat_type.ty).clone(), true),
                };

                params.push(Parameter {
                    name,
                    ty,
                    description,
                    required,
                });
            }
        }
    }

    Ok((context_param, params))
}

fn validate_parameter_descriptions(
    params: &[Parameter],
    param_descriptions: &std::collections::HashMap<String, String>,
    sig: &syn::Signature,
) -> Result<()> {
    let actual_param_names: std::collections::HashSet<String> =
        params.iter().map(|p| p.name.clone()).collect();

    // Check for docstring parameters that don't exist in the function
    for docstring_param in param_descriptions.keys() {
        if !actual_param_names.contains(docstring_param) {
            return Err(syn::Error::new_spanned(
                sig,
                format!(
                    "Parameter '{docstring_param}' found in docstring but not in function parameters"
                ),
            ));
        }
    }

    // Check for missing parameter descriptions in docstring
    for param in params {
        if !param_descriptions.contains_key(&param.name) {
            return Err(syn::Error::new_spanned(
                sig,
                format!(
                    "Parameter '{}' is missing description in docstring. Add: '{}: description'",
                    param.name, param.name
                ),
            ));
        }
    }

    Ok(())
}

fn generate_parameter_struct(params_name: &syn::Ident, params: &[Parameter]) -> TokenStream {
    let fields = params.iter().map(|param| {
        let name = quote::format_ident!("{}", param.name);
        let parameter_type = &param.ty;
        let description = param.description.as_deref().unwrap_or_default();
        let owned_type = if is_str_reference(parameter_type) {
            quote! { ::std::string::String }
        } else {
            quote! { #parameter_type }
        };
        let ty = if param.required {
            quote! { #owned_type }
        } else {
            quote! { ::std::option::Option<#owned_type> }
        };

        quote! {
            #[doc = #description]
            #name: #ty
        }
    });

    quote! {
        #[derive(
            rsai::__private::serde::Deserialize,
            rsai::__private::schemars::JsonSchema
        )]
        #[serde(crate = "rsai::__private::serde", deny_unknown_fields)]
        #[schemars(crate = "rsai::__private::schemars", deny_unknown_fields)]
        struct #params_name {
            #(#fields,)*
        }
    }
}

fn is_str_reference(ty: &Type) -> bool {
    let Type::Reference(reference) = ty else {
        return false;
    };
    let Type::Path(path) = reference.elem.as_ref() else {
        return false;
    };
    path.path.is_ident("str")
}

#[derive(Clone, Copy)]
enum ContextAccess {
    Borrowed,
    OwnedArc,
}

fn generate_execute_impl(
    fn_name: &syn::Ident,
    context_param: &Option<ContextParam>,
    params: &[Parameter],
    params_name: &syn::Ident,
    is_async: bool,
    context_access: ContextAccess,
) -> Result<TokenStream> {
    // Build the function call arguments
    // If there's a context param, it comes first: fn_name(ctx.as_ref(), param1, param2, ...)
    // Otherwise just: fn_name(param1, param2, ...)
    let param_names: Vec<_> = params
        .iter()
        .map(|p| quote::format_ident!("{}", p.name))
        .collect();
    let function_args: Vec<_> = params
        .iter()
        .zip(&param_names)
        .map(|(param, name)| {
            if is_str_reference(&param.ty) {
                if param.required {
                    quote! { #name.as_str() }
                } else {
                    quote! { #name.as_deref() }
                }
            } else {
                quote! { #name }
            }
        })
        .collect();

    let function_call = if let Some(ctx) = context_param {
        let ctx_name = quote::format_ident!("{}", ctx.name);
        if is_async {
            quote! { #fn_name(#ctx_name, #(#function_args),*).await }
        } else {
            quote! { #fn_name(#ctx_name, #(#function_args),*) }
        }
    } else if is_async {
        quote! { #fn_name(#(#function_args),*).await }
    } else {
        quote! { #fn_name(#(#function_args),*) }
    };

    // Generate context extraction if needed
    let context_extraction = if let Some(ctx) = context_param {
        let ctx_name = quote::format_ident!("{}", ctx.name);
        let ctx_inner_ty = &ctx.inner_ty;
        match context_access {
            ContextAccess::Borrowed => quote! {
                let #ctx_name = rsai::Ctx(::std::convert::AsRef::<#ctx_inner_ty>::as_ref(__ctx));
            },
            ContextAccess::OwnedArc => quote! {
                let #ctx_name = rsai::Ctx(::std::convert::AsRef::<#ctx_inner_ty>::as_ref(__ctx.as_ref()));
            },
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        let #params_name { #(#param_names),* } =
            rsai::__private::serde_json::from_value(params).map_err(|e| {
                LlmError::ToolExecution {
                    message: format!("Invalid parameters for {}: {}", stringify!(#fn_name), e),
                    source: Some(Box::new(e)),
                }
            })?;

        #context_extraction

        // Call the function
        let result = #function_call;

        // Convert result to JSON
        rsai::__private::serde_json::to_value(result)
            .map_err(|e| LlmError::ToolExecution {
                message: "Failed to serialize result".to_string(),
                source: Some(Box::new(e)),
            })
    })
}
