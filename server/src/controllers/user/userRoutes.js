import express from "express";
import { upload } from "../../middlewares/multer.middleware.js";

const userRoute = express.Router();

import {
    loginUser,
    createUser,
    getUser,
    logoutUser,
    sendEmailOTP,
    verifyEmailOTP,
} from "./userController.js";
import { verifyJWT } from "../../middlewares/auth.middleware.js";

userRoute.get("/", (req, res) => {
    res.send("User details fetched");
});

userRoute.post("/send-otp", sendEmailOTP);
userRoute.post("/verify-otp", verifyEmailOTP);
userRoute.post("/create", createUser); // For Google registration
userRoute.post("/login", loginUser);

userRoute.patch("/details", (req, res) => {
    res.send("User details updated");
});
userRoute.get("/get_user", verifyJWT, getUser);
userRoute.post("/logout", verifyJWT, logoutUser);
userRoute.post("/admin_login")

export { userRoute };
