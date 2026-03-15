CREATE TABLE customers (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    email       TEXT    UNIQUE,
    phone       TEXT,
    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE products (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    category    TEXT    NOT NULL,  -- 'burger', 'side', 'drink', 'dessert'
    price       REAL    NOT NULL CHECK (price > 0),
    available   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE orders (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id  INTEGER NOT NULL REFERENCES customers(id),
    status       TEXT    NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'preparing', 'ready', 'completed', 'cancelled')),
    total_amount REAL    NOT NULL DEFAULT 0,
    ordered_at   TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE order_items (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id    INTEGER NOT NULL REFERENCES orders(id),
    product_id  INTEGER NOT NULL REFERENCES products(id),
    quantity    INTEGER NOT NULL DEFAULT 1 CHECK (quantity > 0),
    unit_price  REAL    NOT NULL CHECK (unit_price > 0)
);

-- Sample queries

-- Find all pending orders with their customer names and order totals.

-- List the top 5 most expensive products that are currently available, sorted by price descending.

-- Calculate the total revenue per product category from completed orders.
