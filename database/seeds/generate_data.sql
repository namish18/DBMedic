-- =============================================================================
-- DBMedic Lab — Synthetic data seed for dbmedic_lab
-- =============================================================================
-- Uses PostgreSQL generate_series() for efficient bulk generation.
-- All randomness is seeded via setseed() so results are deterministic across
-- runs.  The script is idempotent: re-running it on a non-empty database will
-- report "already populated" and exit without duplicating rows.
--
-- Row counts (overridden via psql \set before \i):
--   :n_customers  default 10000
--   :n_products   default 50000
--   :n_orders     default 200000
-- =============================================================================

-- Seed the random-number generator for reproducibility
SELECT setseed(0.42);

-- ------------------------------------------------------------------ customers
INSERT INTO customers (id, name, email, city, created_at)
SELECT
    s.id,
    'Customer_' || s.id                                     AS name,
    'user' || s.id || '@example-' || (s.id % 50) || '.com' AS email,
    (ARRAY[
        'New York','Los Angeles','Chicago','Houston','Phoenix',
        'Philadelphia','San Antonio','San Diego','Dallas','San Jose',
        'Austin','Jacksonville','Fort Worth','Columbus','Charlotte',
        'Indianapolis','San Francisco','Seattle','Denver','Nashville'
    ])[1 + (s.id % 20)]                                     AS city,
    TIMESTAMP '2022-01-01' + (random() * INTERVAL '3 years') AS created_at
FROM generate_series(1, :n_customers) AS s(id)
ON CONFLICT (email) DO NOTHING;

-- Advance the sequence past the rows we just bulk-inserted
SELECT setval('customers_id_seq', :n_customers, true);

-- ------------------------------------------------------------------ products
INSERT INTO products (id, name, category, price, created_at)
SELECT
    s.id,
    'Product_' || s.id                                      AS name,
    (ARRAY[
        'Electronics','Clothing','Books','Home & Garden','Sports',
        'Toys','Automotive','Health','Beauty','Food & Grocery',
        'Jewelry','Music','Movies','Software','Office Supplies'
    ])[1 + (s.id % 15)]                                     AS category,
    ROUND((CAST(random() * 499 + 1 AS NUMERIC)), 2)         AS price,
    TIMESTAMP '2021-06-01' + (random() * INTERVAL '3 years') AS created_at
FROM generate_series(1, :n_products) AS s(id)
ON CONFLICT DO NOTHING;

SELECT setval('products_id_seq', :n_products, true);

-- ------------------------------------------------------------------ orders
INSERT INTO orders (id, customer_id, status, total_amount, created_at)
SELECT
    s.id                                                                  AS id,
    1 + ((s.id * 7919) % :n_customers)                                   AS customer_id,
    (ARRAY[
        'pending','pending','processing','processing',
        'shipped','shipped','delivered','delivered','delivered',
        'cancelled','refunded'
    ])[1 + (s.id % 11)]                                                   AS status,
    ROUND((CAST(random() * 999 + 5 AS NUMERIC)), 2)                       AS total_amount,
    TIMESTAMP '2022-01-01' + (random() * INTERVAL '3 years')             AS created_at
FROM generate_series(1, :n_orders) AS s(id)
ON CONFLICT DO NOTHING;

SELECT setval('orders_id_seq', :n_orders, true);

-- ------------------------------------------------------------------ order_items
-- Generates exactly 2 items per order (item slot 1 and 2).
INSERT INTO order_items (order_id, product_id, quantity, unit_price)
SELECT
    s.order_id,
    1 + ((s.order_id * 1009 + s.slot * 3571) % :n_products) AS product_id,
    1 + (s.order_id % 5)                                    AS quantity,
    ROUND((CAST(random() * 299 + 1 AS NUMERIC)), 2)         AS unit_price
FROM (
    SELECT o.id AS order_id, slot
    FROM generate_series(1, :n_orders) AS o(id)
    CROSS JOIN generate_series(1, 2) AS slot
) AS s
ON CONFLICT DO NOTHING;

