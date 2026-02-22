use proc_macro2::TokenStream;
use quote::quote;
use syn::{Ident, Result, Token, Type, parse::Parse, parse::ParseStream};

/// Parses a toolset definition with optional context type.
/// Syntax:
/// - `toolset![tool1, tool2]` - context-free toolset
/// - `toolset![ContextType => tool1, tool2]` - toolset with context type
struct ToolsList {
    context_type: Option<Type>,
    tools: Vec<Ident>,
}

impl Parse for ToolsList {
    fn parse(input: ParseStream) -> Result<Self> {
        // Try to parse "Type =>" prefix for context-aware toolsets
        let context_type = if input.peek2(Token![=>])
            || (input.peek(syn::Ident) && {
                // Look ahead to check if it's a type followed by =>
                let fork = input.fork();
                fork.parse::<Type>().is_ok() && fork.peek(Token![=>])
            }) {
            let ty = input.parse::<Type>()?;
            input.parse::<Token![=>]>()?;
            Some(ty)
        } else {
            None
        };

        // Parse comma-separated tool names
        let mut tools = Vec::new();
        while !input.is_empty() {
            tools.push(input.parse::<Ident>()?);

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            } else {
                break;
            }
        }

        Ok(ToolsList {
            context_type,
            tools,
        })
    }
}

pub fn tools_impl(input: TokenStream) -> Result<TokenStream> {
    let tools_list = syn::parse2::<ToolsList>(input)?;

    if tools_list.tools.is_empty() {
        return Err(syn::Error::new(
            proc_macro2::Span::call_site(),
            "toolset! macro requires at least one tool function",
        ));
    }

    // Use the same naming logic as the #[tool] macro to reference existing wrapper structs
    let wrapper_names: Vec<_> = tools_list
        .tools
        .iter()
        .map(|tool_name| {
            quote::format_ident!(
                "{}Tool",
                crate::common::to_pascal_case(&tool_name.to_string())
            )
        })
        .collect();

    // Generate different code based on whether context is present
    let expanded = if let Some(ctx_type) = tools_list.context_type {
        // Context-aware toolset: returns ToolSetBuilder<Ctx> that requires .with_context(ctx)
        quote! {
            {
                use rsai::{ToolFunction, ToolSetBuilder};

                let mut builder = ToolSetBuilder::<#ctx_type>::new();
                #(
                    builder = builder.add_tool(std::sync::Arc::new(#wrapper_names));
                )*
                builder
            }
        }
    } else {
        // Context-free toolset: returns ToolSet<()> directly (backward compatible)
        quote! {
            {
                use rsai::{Tool, ToolChoice, ToolFunction, ToolRegistry, ToolSet};

                let registry = ToolRegistry::new();
                #(
                    registry.register(std::sync::Arc::new(#wrapper_names))
                    .expect(&format!("Failed to register tool: {}", stringify!(#wrapper_names)));
                )*

                ToolSet {
                    registry,
                }
            }
        }
    };

    Ok(expanded)
}
