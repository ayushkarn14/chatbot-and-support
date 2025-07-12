-- Users table (base table for authentication)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    user_type VARCHAR(20) NOT NULL CHECK (user_type IN ('company', 'vendor', 'admin')),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Companies table
CREATE TABLE companies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20) UNIQUE NOT NULL,
    address TEXT,
    executives JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Vendors table
CREATE TABLE vendors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20) UNIQUE NOT NULL,
    address TEXT,
    type TEXT[],
    rating DECIMAL(3,2) CHECK (rating >= 0 AND rating <= 5),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Requirements table
CREATE TABLE requirements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    company_id UUID REFERENCES companies(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    workforce_size INT NOT NULL CHECK (workforce_size > 0),
    job_category VARCHAR(50) NOT NULL,
    employment_type VARCHAR(20) NOT NULL CHECK (employment_type IN ('Permanent', 'On Demand')),
    background_check BOOLEAN DEFAULT false,
    language_known VARCHAR(100),
    vehicle_required VARCHAR(50),
    location TEXT NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    coordinates JSONB,
    max_distance INTEGER NOT NULL,
    total_hours INTEGER,
    compensation DECIMAL(10,2),
    estimated_payout DECIMAL(10,2) NOT NULL,
    special_requirements TEXT,
    medical_check_required BOOLEAN DEFAULT false,
    medical_requirements_description TEXT,
    status VARCHAR(20) CHECK (status IN ('pending', 'active', 'completed', 'cancelled')) DEFAULT 'pending',
    start_date DATE,
    end_date DATE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT valid_dates CHECK (
        (employment_type = 'Permanent' AND start_date IS NULL AND end_date IS NULL) OR
        (employment_type = 'On Demand' AND start_date IS NOT NULL AND end_date IS NOT NULL AND end_date >= start_date)
    )
);

-- Workers table
CREATE TABLE workers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id UUID REFERENCES vendors(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    phone VARCHAR(20) UNIQUE NOT NULL,
    experience_years INT,
    status VARCHAR(20) CHECK (status IN ('available', 'assigned', 'unavailable')) DEFAULT 'available',
    current_requirement_id UUID REFERENCES requirements(id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Workforce table
CREATE TABLE workforce (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    vendor_id UUID REFERENCES vendors(id) ON DELETE CASCADE,
    phone VARCHAR(20) NOT NULL,
    address TEXT NOT NULL,
    pincode VARCHAR(6) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(100) NOT NULL,
    number_of_workers INTEGER NOT NULL CHECK (number_of_workers > 0),
    job_category VARCHAR(50) NOT NULL,
    employment_type VARCHAR(20) NOT NULL CHECK (employment_type IN ('Permanent', 'On Demand')),
    availability VARCHAR(20) NOT NULL CHECK (availability IN ('fullTime', 'partTime', 'onDemand')),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'accepted', 'rejected')),
    location_name VARCHAR(100) NOT NULL,
    coordinates JSONB,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Negotiations table
CREATE TABLE negotiations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requirement_id UUID REFERENCES requirements(id) ON DELETE CASCADE,
    workforce_id UUID REFERENCES workforce(id) ON DELETE CASCADE,
    vendor_id UUID REFERENCES vendors(id),
    company_phone VARCHAR(20) NOT NULL,
    vendor_phone VARCHAR(20) NOT NULL,
    status VARCHAR(20) CHECK (status IN ('pending', 'accepted', 'rejected', 'cancelled')) DEFAULT 'pending',
    original_price DECIMAL(10,2) NOT NULL,
    quoted_price DECIMAL(10,2),
    vendor_quoted_price DECIMAL(10,2),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_requirement_workforce UNIQUE (requirement_id, workforce_id)
);

-- Accepted Tasks table
CREATE TABLE accepted_tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requirement_id UUID REFERENCES requirements(id) ON DELETE CASCADE,
    workforce_id UUID REFERENCES workforce(id) ON DELETE CASCADE,
    company_phone VARCHAR(20) NOT NULL,
    vendor_phone VARCHAR(20) NOT NULL,
    final_price DECIMAL(10,2) NOT NULL,
    accepted_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT unique_requirement_acceptance UNIQUE (requirement_id)
);
-- Chat history table
CREATE TABLE chat_history (
    id SERIAL PRIMARY KEY,                  
    session_id VARCHAR(36) NOT NULL,        
    turn INTEGER NOT NULL,            
    user_query TEXT NOT NULL,             
    assistant_answer TEXT NOT NULL,       
    timestamp TIMESTAMPTZ DEFAULT NOW(),   

    CONSTRAINT chat_history_session_turn_idx UNIQUE (session_id, turn)
);
