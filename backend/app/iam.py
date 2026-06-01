import time
import base64
import xml.etree.ElementTree as ET
from typing import List, Optional
from pydantic import BaseModel
from jose import jwt, JWTError

# Secret key for signing simulated JWTs
SECRET_KEY = "langgraph-iam-secret-key"
ALGORITHM = "HS256"

class SecurityContext(BaseModel):
    user_id: str
    username: str
    email: str
    role: str          # guest, employee, manager, admin
    department: str    # IT, HR, Engineering, Finance, Public
    scopes: List[str]  # read:public, read:restricted, write:all, etc.
    auth_method: str   # OAuth2, OIDC, SAML
    token_raw: str     # Raw token string for inspection

# --- SIMULATED USER DATABASE ---
PERSONAS = {
    "guest_user": {
        "user_id": "usr_9988",
        "username": "guest_anon",
        "email": "guest@company.com",
        "role": "guest",
        "department": "Public",
        "scopes": ["read:public"]
    },
    "employee_hr": {
        "user_id": "usr_4432",
        "username": "sarah_hr",
        "email": "sarah.jones@company.com",
        "role": "employee",
        "department": "HR",
        "scopes": ["read:public", "read:restricted", "read:hr"]
    },
    "it_admin": {
        "user_id": "usr_0001",
        "username": "alex_admin",
        "email": "alex.sysadmin@company.com",
        "role": "admin",
        "department": "IT",
        "scopes": ["read:public", "read:restricted", "read:hr", "write:system"]
    }
}

# --- OAUTH 2.0 & OIDC GENERATORS ---
def generate_oauth_jwt(username: str, expires_in: int = 3600) -> str:
    """Generates an OAuth 2.0 Access Token (JWT) with scopes."""
    user = PERSONAS.get(username)
    if not user:
        raise ValueError("User persona not found")
    
    payload = {
        "sub": user["user_id"],
        "exp": time.time() + expires_in,
        "scope": " ".join(user["scopes"]),
        "client_id": "enterprise_chat_app",
        "iss": "https://auth.company.com"
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def generate_oidc_jwt(username: str, expires_in: int = 3600) -> str:
    """Generates an OIDC ID Token (JWT) containing profile claims and roles."""
    user = PERSONAS.get(username)
    if not user:
        raise ValueError("User persona not found")
    
    payload = {
        "sub": user["user_id"],
        "iss": "https://auth.company.com/oidc",
        "exp": time.time() + expires_in,
        "name": user["username"],
        "email": user["email"],
        "role": user["role"],
        "department": user["department"]
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

# --- SAML 2.0 XML GENERATOR & PARSER ---
# SAML uses XML assertions. We will simulate a signed SAML Assertion and parse it.
def generate_saml_assertion(username: str) -> str:
    """Generates a base64 encoded mock SAML 2.0 Assertion XML."""
    user = PERSONAS.get(username)
    if not user:
        raise ValueError("User persona not found")
    
    saml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<saml2:Assertion ID="_a1b2c3d4e5f6g7h8i9" IssueInstant="2026-05-29T12:00:00Z" Version="2.0"
    xmlns:saml2="urn:oasis:names:tc:SAML:2.0:assertion">
    <saml2:Issuer>https://pingidentity.company.com/idp</saml2:Issuer>
    <saml2:Subject>
        <saml2:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">{user["email"]}</saml2:NameID>
    </saml2:Subject>
    <saml2:AttributeStatement>
        <saml2:Attribute Name="UserID">
            <saml2:AttributeValue>{user["user_id"]}</saml2:AttributeValue>
        </saml2:Attribute>
        <saml2:Attribute Name="Username">
            <saml2:AttributeValue>{user["username"]}</saml2:AttributeValue>
        </saml2:Attribute>
        <saml2:Attribute Name="Role">
            <saml2:AttributeValue>{user["role"]}</saml2:AttributeValue>
        </saml2:Attribute>
        <saml2:Attribute Name="Department">
            <saml2:AttributeValue>{user["department"]}</saml2:AttributeValue>
        </saml2:Attribute>
        <saml2:Attribute Name="Scopes">
            <saml2:AttributeValue>{",".join(user["scopes"])}</saml2:AttributeValue>
        </saml2:Attribute>
    </saml2:AttributeStatement>
    <Signature xmlns="http://www.w3.org/2000/09/xmldsig#">
        <!-- Mock Digital Signature to represent SAML trust validation -->
        <SignatureValue>MOCK_SIGNATURE_VALID_TRUST_OK_PING_IDENTITY</SignatureValue>
    </Signature>
</saml2:Assertion>"""
    return base64.b64encode(saml_xml.encode("utf-8")).decode("utf-8")

# --- UNIFIED AUTHENTICATION GATEWAY ---
def authenticate_request(auth_header: str) -> SecurityContext:
    """
    Decodes the authorization header (OAuth2 JWT, OIDC JWT, or SAML Assertion)
    and maps it to a unified SecurityContext object.
    """
    if not auth_header:
        raise ValueError("Missing Authorization Header")
        
    parts = auth_header.split()
    if len(parts) != 2:
        raise ValueError("Invalid Authorization Header format. Must be 'Bearer <token>' or 'SAML <assertion>'")
    
    scheme, token = parts
    
    if scheme.lower() == "bearer":
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            # Determine if it's OIDC (has profile info like role/department) or OAuth 2.0 (has scopes)
            if "role" in payload:
                # OIDC Flow
                # For OIDC, scopes are usually implicit or mapped from roles
                role = payload.get("role", "guest")
                department = payload.get("department", "Public")
                scopes = ["read:public"]
                if role in ["employee", "manager"]:
                    scopes.extend(["read:restricted"])
                if role == "admin":
                    scopes.extend(["read:restricted", "read:hr", "write:system"])
                if department == "HR":
                    scopes.append("read:hr")
                
                return SecurityContext(
                    user_id=payload["sub"],
                    username=payload.get("name", "unknown"),
                    email=payload.get("email", "unknown@company.com"),
                    role=role,
                    department=department,
                    scopes=scopes,
                    auth_method="OIDC ID Token (JWT)",
                    token_raw=token
                )
            else:
                # Standard OAuth 2.0 Access Token Flow
                scopes_str = payload.get("scope", "read:public")
                scopes = scopes_str.split()
                
                # Fetch identity metadata back from persona database to mimic an introspect/UserInfo endpoint call
                user_id = payload["sub"]
                persona_match = next((p for p in PERSONAS.values() if p["user_id"] == user_id), None)
                
                return SecurityContext(
                    user_id=user_id,
                    username=persona_match["username"] if persona_match else "oauth_client",
                    email=persona_match["email"] if persona_match else "oauth@company.com",
                    role=persona_match["role"] if persona_match else "guest",
                    department=persona_match["department"] if persona_match else "Public",
                    scopes=scopes,
                    auth_method="OAuth 2.0 Access Token (JWT)",
                    token_raw=token
                )
        except JWTError as e:
            raise ValueError(f"Invalid JWT Token: {str(e)}")
            
    elif scheme.lower() == "saml":
        try:
            xml_bytes = base64.b64decode(token)
            root = ET.fromstring(xml_bytes)
            
            # Simple XPath mapping to extract attributes from SAML Assertion
            ns = {"saml2": "urn:oasis:names:tc:SAML:2.0:assertion"}
            
            # Get Subject email
            name_id = root.find(".//saml2:NameID", ns)
            email = name_id.text if name_id is not None else "unknown@company.com"
            
            # Extract Attributes
            attrs = {}
            for attr in root.findall(".//saml2:Attribute", ns):
                name = attr.get("Name")
                val_node = attr.find("saml2:AttributeValue", ns)
                if val_node is not None:
                    attrs[name] = val_node.text
            
            scopes = attrs.get("Scopes", "read:public").split(",")
            
            # Check signature presence
            sig = root.find("{http://www.w3.org/2000/09/xmldsig#}Signature")
            if sig is None:
                raise ValueError("SAML signature verification failed: Signature element missing")
                
            return SecurityContext(
                user_id=attrs.get("UserID", "unknown_saml"),
                username=attrs.get("Username", "unknown_saml"),
                email=email,
                role=attrs.get("Role", "guest"),
                department=attrs.get("Department", "Public"),
                scopes=scopes,
                auth_method="SAML 2.0 Assertion (XML)",
                token_raw=xml_bytes.decode("utf-8")
            )
        except Exception as e:
            raise ValueError(f"Failed to decode SAML assertion: {str(e)}")
            
    else:
        raise ValueError(f"Unsupported authentication scheme: {scheme}")
