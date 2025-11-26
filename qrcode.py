import qrcode
from qrcode.image.styledpil import StyledPilImage
from qrcode.image.styles.moduledrawers import RoundedModuleDrawer

qr = qrcode.QRCode(
    version=1,
    box_size=10,
    border=4
)

qr.add_data("Hello from Python!")
qr.make(fit=True)

img = qr.make_image(
    fill_color="blue",
    back_color="white",
    image_factory=StyledPilImage,
    module_drawer=RoundedModuleDrawer()
)

img.save("styled_qr.png")
print("Styled QR Code Created!")
