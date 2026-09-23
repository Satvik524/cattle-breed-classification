from fastapi import UploadFile, File, HTTPException


def validate_image(
        file: UploadFile=File(...)
):
    allowed_type={
        'image/jpeg',
        'image/png'
    }

    if file.content_type not in allowed_type:
        raise HTTPException(status_code = 400 , detail="Only JPG, JPEG, and PNG images are allowed.")

    return file