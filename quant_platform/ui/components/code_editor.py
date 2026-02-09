"""
Code editor component using streamlit-ace.
Provides Python code editing with syntax highlighting.
"""

import streamlit as st
from typing import Optional, Callable


def render_code_editor(
    default_code: str = "",
    key: str = "code_editor",
    height: int = 400,
    language: str = "python",
    theme: str = "monokai",
    font_size: int = 14,
    show_gutter: bool = True,
    show_print_margin: bool = False,
    wrap: bool = True,
    readonly: bool = False,
    on_change: Optional[Callable] = None,
) -> str:
    """
    Render a code editor component.
    
    Args:
        default_code: Initial code content
        key: Unique key for the component
        height: Editor height in pixels
        language: Syntax highlighting language
        theme: Editor theme (monokai, github, tomorrow, etc.)
        font_size: Font size in pixels
        show_gutter: Show line numbers
        show_print_margin: Show print margin line
        wrap: Enable line wrapping
        readonly: Make editor read-only
        on_change: Callback when code changes
        
    Returns:
        Current code content
    """
    try:
        from streamlit_ace import st_ace
        
        code = st_ace(
            value=default_code,
            language=language,
            theme=theme,
            height=height,
            font_size=font_size,
            show_gutter=show_gutter,
            show_print_margin=show_print_margin,
            wrap=wrap,
            readonly=readonly,
            key=key,
            auto_update=True,
        )
        
        return code or default_code
        
    except ImportError:
        # Fallback to text_area if streamlit-ace not installed
        st.warning("⚠️ 代码编辑器组件未安装，使用基础文本框。请运行: `pip install streamlit-ace`")
        
        code = st.text_area(
            "代码编辑器",
            value=default_code,
            height=height,
            key=key,
        )
        
        return code


def render_code_editor_with_toolbar(
    default_code: str = "",
    key: str = "code_editor_toolbar",
    height: int = 400,
    templates: dict = None,
) -> str:
    """
    Render code editor with toolbar for templates and actions.
    
    Args:
        default_code: Initial code
        key: Component key
        height: Editor height
        templates: Dictionary of template name -> code
        
    Returns:
        Current code content
    """
    # Initialize session state for code
    state_key = f"{key}_code"
    editor_version_key = f"{key}_editor_version"
    
    if state_key not in st.session_state:
        st.session_state[state_key] = default_code
    
    # Initialize editor version (used to force re-render)
    if editor_version_key not in st.session_state:
        st.session_state[editor_version_key] = 0
    
    # Toolbar
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        if templates:
            template_names = ["选择模板..."] + list(templates.keys())
            selected_template = st.selectbox(
                "策略模板",
                template_names,
                key=f"{key}_template_select",
                label_visibility="collapsed",
            )
            
            if selected_template != "选择模板..." and selected_template in templates:
                if st.session_state.get(f"{key}_last_template") != selected_template:
                    st.session_state[state_key] = templates[selected_template]
                    st.session_state[f"{key}_last_template"] = selected_template
                    # Increment version to force editor re-render with new key
                    st.session_state[editor_version_key] += 1
                    st.rerun()
    
    with col2:
        theme_options = {
            "Monokai (深色)": "monokai",
            "GitHub (浅色)": "github",
            "Tomorrow": "tomorrow",
            "Twilight": "twilight",
            "XCode": "xcode",
        }
        theme_name = st.selectbox(
            "主题",
            list(theme_options.keys()),
            key=f"{key}_theme",
            label_visibility="collapsed",
        )
        theme = theme_options[theme_name]
    
    with col3:
        font_size = st.selectbox(
            "字号",
            [12, 14, 16, 18],
            index=1,
            key=f"{key}_font",
            label_visibility="collapsed",
        )
    
    with col4:
        if st.button("🗑️ 清空", key=f"{key}_clear", use_container_width=True):
            st.session_state[state_key] = ""
            # Increment version to force editor re-render
            st.session_state[editor_version_key] += 1
            st.rerun()
    
    # Editor - use versioned key to force re-render when template changes
    editor_key = f"{key}_editor_v{st.session_state[editor_version_key]}"
    
    code = render_code_editor(
        default_code=st.session_state[state_key],
        key=editor_key,
        height=height,
        theme=theme,
        font_size=font_size,
    )
    
    # Update session state
    st.session_state[state_key] = code
    
    return code


def render_code_viewer(
    code: str,
    language: str = "python",
    height: int = 300,
    title: str = "",
):
    """
    Render a read-only code viewer.
    
    Args:
        code: Code to display
        language: Syntax highlighting language
        height: Viewer height
        title: Optional title
    """
    if title:
        st.markdown(f"**{title}**")
    
    render_code_editor(
        default_code=code,
        key=f"viewer_{hash(code)}",
        height=height,
        language=language,
        readonly=True,
        theme="github",
    )
