"""
Jinja2渲染工具，用于生成新的策略和配置。
"""


def render_template(templatefile: str, arguments: dict) -> str:
    from jinja2 import Environment, PackageLoader, select_autoescape

    env = Environment(
        loader=PackageLoader("freqtrade", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(templatefile)
    return template.render(**arguments)


def render_template_with_fallback(
    templatefile: str, templatefallbackfile: str, arguments: dict | None = None
) -> str:
    """
    如果可能则使用templatefile，否则回退到templatefallbackfile
    """
    from jinja2.exceptions import TemplateNotFound

    if arguments is None:
        arguments = {}
    try:
        return render_template(templatefile, arguments)
    except TemplateNotFound:
        return render_template(templatefallbackfile, arguments)