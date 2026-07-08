import reflex as rx


config = rx.Config(
    app_name="legal_ui",
    frontend_port=3000,
    backend_port=8001,
    plugins=[
        rx.plugins.SitemapPlugin(),
        rx.plugins.RadixThemesPlugin(
            theme=rx.theme(
                appearance="light",
                accent_color="indigo",
                radius="small",
            )
        ),
    ],
)
