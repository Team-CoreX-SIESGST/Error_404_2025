import User from "../../models/user.js";
import bcrypt from "bcrypt";
import jwt from "jsonwebtoken";
import crypto from "crypto";
import {
    asyncHandler,
    uploadOnCloudinary,
    deleteOnCloudinary,
    statusType,
    sendResponse
} from "../../utils/index.js";
import { verifyGoogleToken } from "../../utils/googleAuth.js";
import { sendEmail } from "../../utils/emailService.js"; // Adjust the path as needed

// Token generator functions
const generateAccessToken = (user) => {
    return jwt.sign(
        { user_id: user._id, date_of_birth: user.date_of_birth },
        process.env.ACCESS_TOKEN_SECRET,
        { expiresIn: "1d" }
    );
};

const generateRefreshToken = (user) => {
    return jwt.sign(
        {
            user_id: user._id,
            date_of_birth: user.date_of_birth,
            token_version: user.token_version || 0
        },
        process.env.REFRESH_TOKEN_SECRET,
        { expiresIn: "7d" }
    );
};

const cookieOptions = {
    httpOnly: false,
    secure: process.env.NODE_ENV === "production",
    sameSite: "Strict"
};

// Generate verification token
const generateVerificationToken = () => {
    return crypto.randomBytes(32).toString("hex");
};

// Generate OTP (6 digits)
const generateOTP = () => {
    return Math.floor(100000 + Math.random() * 900000).toString();
};

// Send OTP for email verification using your sendEmail function
export const sendEmailOTP = asyncHandler(async (req, res) => {
    const { email } = req.body;

    if (!email) {
        return sendResponse(res, false, null, "Email is required", statusType.BAD_REQUEST);
    }

    // Check if user already exists
    const existingUser = await User.findOne({ email, isVerified: true });
    if (existingUser) {
        return sendResponse(
            res,
            false,
            null,
            "User already exists with this email",
            statusType.BAD_REQUEST
        );
    }

    // Generate OTP and verification token
    const otp = generateOTP();
    const verificationToken = generateVerificationToken();
    const verificationExpires = new Date(Date.now() + 10 * 60 * 1000); // 10 minutes

    // Find or create unverified user
    let user = await User.findOne({ email, isVerified: false });

    if (user) {
        user.verificationToken = verificationToken;
        user.verificationExpires = verificationExpires;
        user.otp = otp; // Store OTP for verification
        await user.save();
    } else {
        user = await User.create({
            email,
            verificationToken,
            verificationExpires,
            otp,
            isVerified: false
        });
    }

    // Send OTP email using your sendEmail function
    const emailSubject = "Your OTP for Email Verification";
    const emailBody = `
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                .container { max-width: 600px; margin: 0 auto; padding: 20px; font-family: Arial, sans-serif; }
                .header { background: linear-gradient(135deg, #10b981, #059669); padding: 20px; text-align: center; color: white; border-radius: 10px 10px 0 0; }
                .content { background: #f9fafb; padding: 30px; border-radius: 0 0 10px 10px; }
                .otp-code { font-size: 32px; font-weight: bold; text-align: center; color: #059669; margin: 20px 0; }
                .footer { text-align: center; margin-top: 20px; color: #6b7280; font-size: 14px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Email Verification</h1>
                </div>
                <div class="content">
                    <h2>Hello,</h2>
                    <p>Thank you for registering. Use the OTP below to verify your email address:</p>
                    <div class="otp-code">${otp}</div>
                    <p>This OTP will expire in 10 minutes.</p>
                    <p>If you didn't request this, please ignore this email.</p>
                </div>
                <div class="footer">
                    <p>&copy; ${new Date().getFullYear()} Your App Name. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
    `;

    try {
        await sendEmail(
            process.env.SENDER_EMAIL, // from
            email, // to
            emailSubject, // subject
            emailBody // body
        );

        return sendResponse(res, true, { verificationToken }, "OTP sent to email", statusType.OK);
    } catch (error) {
        console.error("Error sending OTP email:", error);
        return sendResponse(
            res,
            false,
            null,
            "Failed to send OTP email",
            statusType.INTERNAL_SERVER_ERROR
        );
    }
});

// Verify OTP and complete basic registration
export const verifyEmailOTP = asyncHandler(async (req, res) => {
    const { email, otp, verificationToken, password, name, date_of_birth } = req.body;

    if (!email || !otp || !verificationToken) {
        return sendResponse(
            res,
            false,
            null,
            "Email, OTP and verification token are required",
            statusType.BAD_REQUEST
        );
    }

    // Find unverified user
    const user = await User.findOne({
        email,
        verificationToken,
        verificationExpires: { $gt: new Date() }
    });

    if (!user) {
        return sendResponse(res, false, null, "Invalid or expired OTP", statusType.BAD_REQUEST);
    }
    console.log("user", user.otp);
    // Verify OTP
    if (user.otp !== otp) {
        return sendResponse(res, false, null, "Invalid OTP", statusType.BAD_REQUEST);
    }

    // Update user with basic details
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    user.name = name;
    user.date_of_birth = date_of_birth;
    user.password = hashedPassword;
    user.isVerified = true;
    user.verificationToken = undefined;
    user.verificationExpires = undefined;
    user.otp = undefined;

    await user.save();

    // Assign free plan
    await assignFreePlan(user._id);

    // Generate tokens
    const accessToken = generateAccessToken(user);
    const refreshToken = generateRefreshToken(user);

    user.refresh_token = refreshToken;
    await user.save();

    const userData = user.toObject();
    delete userData.password;

    res.cookie("accessToken", accessToken, cookieOptions);
    res.cookie("refreshToken", refreshToken, cookieOptions);

    return sendResponse(
        res,
        true,
        {
            ...userData,
            accessToken,
            requiresAshaDetails: true // Flag to indicate frontend to collect Asha details
        },
        "Email verified and basic registration completed",
        statusType.OK
    );
});

// Updated createUser for Google registration
export const createUser = asyncHandler(async (req, res) => {
    const { name, email, password, date_of_birth, image, googleToken, role } = req.body;

    // Google Auth Flow
    if (googleToken) {
        console.log(googleToken);
        try {
            const googleUser = await verifyGoogleToken(googleToken);
            console.log(googleUser);
            let user = await User.findOne({
                $or: [{ email: googleUser.email }, { googleId: googleUser.sub }]
            });

            if (user) {
                return sendResponse(
                    res,
                    false,
                    null,
                    "User already exists",
                    statusType.BAD_REQUEST
                );
            }

            // Create new user with Google data
            user = await User.create({
                name: googleUser.name,
                email: googleUser.email,
                googleId: googleUser.sub,
                image: googleUser.picture,
                date_of_birth: date_of_birth || null,
                role: role || "user",
                isVerified: true
            });

            // Assign free plan to new user
            await assignFreePlan(user._id);

            // Generate tokens and respond
            const accessToken = generateAccessToken(user);
            const refreshToken = generateRefreshToken(user);

            user.refresh_token = refreshToken;
            await user.save();

            const userData = user.toObject();
            delete userData.password;

            res.cookie("accessToken", accessToken, cookieOptions);
            res.cookie("refreshToken", refreshToken, cookieOptions);

            return sendResponse(
                res,
                true,
                {
                    ...userData,
                    accessToken,
                    requiresAshaDetails: role === "asha_worker"
                },
                "User registered with Google",
                statusType.CREATED
            );
        } catch (error) {
            return sendResponse(res, false, null, error.message, statusType.BAD_REQUEST);
        }
    } else {
        // Regular email registration - now handled by OTP flow
        return sendResponse(
            res,
            false,
            null,
            "Please use OTP verification for email registration",
            statusType.BAD_REQUEST
        );
    }
});

export const loginUser = asyncHandler(async (req, res) => {
    const { email, password, googleToken } = req.body;
    if (googleToken) {
        try {
            const googleUser = await verifyGoogleToken(googleToken);

            let user = await User.findOne({
                $or: [{ email: googleUser.email }, { googleId: googleUser.sub }]
            });

            // Auto-register if user doesn't exist
            if (!user) {
                user = await User.create({
                    name: googleUser.name,
                    email: googleUser.email,
                    googleId: googleUser.sub,
                    image: googleUser.picture
                });
                // Assign free plan to new user
                await assignFreePlan(user._id);
            }

            // Generate tokens and respond
            const accessToken = generateAccessToken(user);
            const refreshToken = generateRefreshToken(user);

            user.refresh_token = refreshToken;
            await user.save();

            const userData = user.toObject();
            delete userData.password;

            res.cookie("accessToken", accessToken, cookieOptions);
            res.cookie("refreshToken", refreshToken, cookieOptions);

            return sendResponse(
                res,
                true,
                { ...userData, accessToken },
                "Login with Google Successful",
                statusType.OK
            );
        } catch (error) {
            return sendResponse(res, false, null, error.message, statusType.BAD_REQUEST);
        }
    } else {
        if (!email || !password) {
            return sendResponse(
                res,
                false,
                null,
                "Email and Password are required",
                statusType.BAD_REQUEST
            );
        }

        const user = await User.findOne({ email, isVerified: true });
        if (!user) {
            return sendResponse(res, false, null, "User does not exist", statusType.BAD_REQUEST);
        }

        const isMatch = await bcrypt.compare(password, user.password);
        if (!isMatch) {
            return sendResponse(
                res,
                false,
                null,
                "Email or Password is incorrect",
                statusType.BAD_REQUEST
            );
        }

        const accessToken = generateAccessToken(user);
        const refreshToken = generateRefreshToken(user);

        user.refresh_token = refreshToken;
        await user.save();

        const userData = user.toObject();
        delete userData.password;

        res.cookie("accessToken", accessToken, cookieOptions);
        res.cookie("refreshToken", refreshToken, cookieOptions);

        return sendResponse(
            res,
            true,
            { ...userData, accessToken },
            "Login Successful",
            statusType.OK
        );
    }
});

export const getUser = asyncHandler(async (req, res) => {
    const user = req.user;

    if (!user) {
        return sendResponse(res, false, null, "User not found", statusType.NOT_FOUND);
    }

    return sendResponse(res, true, user, "User retrieved successfully", statusType.OK);
});

export const logoutUser = asyncHandler(async (req, res) => {
    await User.findByIdAndUpdate(
        req.user._id,
        {
            $unset: { refresh_token: 1 },
            $inc: { token_version: 1 }
        },
        { new: true }
    );

    return sendResponse(res, true, null, "User logged out successfully", statusType.OK);
});
