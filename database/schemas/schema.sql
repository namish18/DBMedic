-- =============================================================================
-- DBMedic Lab — Schema definition for dbmedic_lab
-- =============================================================================
-- This schema is intentionally missing secondary indexes on foreign-key columns
-- (e.g. orders.customer_id) so that slow-query incidents can be reproduced
-- deterministically.  Indexes are added only through the approved remediation
-- path in agent/tools/query_analyzer.py.
-- =============================================================================

-- ------------------------------------------------------------------ customers
CREATE TABLE IF NOT EXISTS customers (
    id         BIGSERIAL    PRIMARY KEY,
    name       VARCHAR(120) NOT NULL,
    email      VARCHAR(254) NOT NULL,
    city       VARCHAR(100) NOT NULL,
    created_at TIMESTAMP    NOT NULL DEFAULT NOW(),
    CONSTRAINT customers_email_unique UNIQUE (email)
);

-- ------------------------------------------------------------------ products
CREATE TABLE IF NOT EXISTS products (
    id         BIGSERIAL     PRIMARY KEY,
    name       VARCHAR(200)  NOT NULL,
    category   VARCHAR(80)   NOT NULL,
    price      NUMERIC(10,2) NOT NULL CHECK (price >= 0),
    created_at TIMESTAMP     NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------------ orders
-- NOTE: No index on customer_id — this is intentional for the slow-query
-- incident scenario.  The remediation script adds idx_orders_customer_id.
CREATE TABLE IF NOT EXISTS orders (
    id           BIGSERIAL     PRIMARY KEY,
    customer_id  BIGINT        NOT NULL REFERENCES customers(id),
    status       VARCHAR(20)   NOT NULL
                     CHECK (status IN ('pending','processing','shipped',
                                       'delivered','cancelled','refunded')),
    total_amount NUMERIC(12,2) NOT NULL CHECK (total_amount >= 0),
    created_at   TIMESTAMP     NOT NULL DEFAULT NOW()
);

-- ------------------------------------------------------------------ order_items
CREATE TABLE IF NOT EXISTS order_items (
    id         BIGSERIAL     PRIMARY KEY,
    order_id   BIGINT        NOT NULL REFERENCES orders(id),
    product_id BIGINT        NOT NULL REFERENCES products(id),
    quantity   INTEGER       NOT NULL CHECK (quantity > 0),
    unit_price NUMERIC(10,2) NOT NULL CHECK (unit_price >= 0)
);

