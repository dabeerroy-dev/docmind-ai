# ============================================
# STRIPE PAYMENT SYSTEM
# Handles subscriptions and billing
# International level payment processing!
# ============================================

# stripe: official Stripe Python library
import stripe

# Import our Stripe configuration
from stripe_config import (
    STRIPE_SECRET_KEY,
    STRIPE_PRICE_ID,
    SUCCESS_URL,
    CANCEL_URL
)

# ============================================
# SETUP STRIPE
# Set secret key for all API calls
# ============================================
stripe.api_key = STRIPE_SECRET_KEY

# ============================================
# FUNCTION 1: Create Checkout Session
# Input: user email + uid
# Output: payment URL to redirect user
# User clicks → goes to Stripe payment page
# ============================================
def create_checkout_session(email, uid):
    try:
        # Create Stripe checkout session
        session = stripe.checkout.Session.create(

            # Payment method types accepted
            payment_method_types=["card"],

            # What they are buying
            line_items=[{
                "price": STRIPE_PRICE_ID,
                "quantity": 1
            }],

            # Subscription mode = recurring monthly
            mode="subscription",

            # Pre-fill email in checkout form
            customer_email=email,

            # Add user ID to track who paid
            metadata={"uid": uid},

            # Where to go after success
            success_url=SUCCESS_URL,

            # Where to go if cancelled
            cancel_url=CANCEL_URL
        )

        print(f"Checkout session created: {session.id}")
        return {
            "success": True,
            "url": session.url,
            "session_id": session.id
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# FUNCTION 2: Check Subscription Status
# Input: user email
# Output: active or not active
# ============================================
def check_subscription(email):
    try:
        # Find customer by email
        customers = stripe.Customer.list(email=email)

        # No customer found = not subscribed
        if not customers.data:
            return {
                "active": False,
                "plan": "free"
            }

        customer = customers.data[0]

        # Get their subscriptions
        subscriptions = stripe.Subscription.list(
            customer=customer.id,
            status="active"
        )

        # Has active subscription = PRO user!
        if subscriptions.data:
            return {
                "active": True,
                "plan": "pro",
                "customer_id": customer.id
            }

        # No active subscription = FREE user
        return {
            "active": False,
            "plan": "free"
        }

    except Exception as e:
        return {
            "active": False,
            "plan": "free",
            "error": str(e)
        }

# ============================================
# FUNCTION 3: Cancel Subscription
# Input: customer email
# Output: cancellation confirmation
# ============================================
def cancel_subscription(email):
    try:
        # Find customer
        customers = stripe.Customer.list(email=email)

        if not customers.data:
            return {
                "success": False,
                "error": "No subscription found!"
            }

        customer = customers.data[0]

        # Get active subscriptions
        subscriptions = stripe.Subscription.list(
            customer=customer.id,
            status="active"
        )

        if not subscriptions.data:
            return {
                "success": False,
                "error": "No active subscription!"
            }

        # Cancel at end of billing period
        # User keeps access until period ends!
        subscription = subscriptions.data[0]
        stripe.Subscription.modify(
            subscription.id,
            cancel_at_period_end=True
        )

        return {
            "success": True,
            "message": "Subscription cancelled!"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# ============================================
# TEST PAYMENT SYSTEM
# ============================================
if __name__ == "__main__":
    print("Testing Stripe connection...")

    # Test creating checkout session
    result = create_checkout_session(
        "test@example.com",
        "test_uid_123"
    )

    if result["success"]:
        print("✅ Stripe connected!")
        print(f"Payment URL: {result['url']}")
    else:
        print(f"❌ Error: {result['error']}")