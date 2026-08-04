-- 1) Save a new pending review (always forces status='pending')
CREATE OR REPLACE FUNCTION public.save_pending_review(
  p_image_path text,
  p_raw_ocr_text text,
  p_extracted_json jsonb,
  p_failure_reasons text[],
  p_status text
)
RETURNS uuid
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
  v_id uuid;
BEGIN
  -- Per your choice (A): ignore p_status and always use 'pending'
  INSERT INTO public.invoice_reviews (
    image_path,
    raw_ocr_text,
    extracted_json,
    failure_reasons,
    status,
    reviewer_comment,
    reviewed_at
  ) VALUES (
    p_image_path,
    p_raw_ocr_text,
    p_extracted_json,
    p_failure_reasons,
    'pending',
    NULL,
    NULL
  )
  RETURNING id INTO v_id;

  RETURN v_id;
END;
$$;


-- 2) Load a pending review by id (returns the full row)
CREATE OR REPLACE FUNCTION public.load_pending_review(
)
RETURNS TABLE (
  id uuid,
  image_path text,
  raw_ocr_text text,
  extracted_json jsonb,
  failure_reasons text[],
  status text,
  reviewer_comment text,
  reviewed_at timestamptz,
  created_at timestamptz
)
LANGUAGE sql
SECURITY INVOKER
AS $$
  SELECT
    ir.id,
    ir.image_path,
    ir.raw_ocr_text,
    ir.extracted_json,
    ir.failure_reasons,
    ir.status,
    ir.reviewer_comment,
    ir.reviewed_at,
    ir.created_at
  FROM public.invoice_reviews ir
  WHERE ir.status = 'pending'
  ORDER BY ir.created_at DESC;
$$;


-- 3) Approve: set status='approved', reviewer_comment, reviewed_at
-- (Only updates if currently pending)
CREATE OR REPLACE FUNCTION public.approve_review(
  p_review_id uuid,
  p_reviewer_comment text
)
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
  UPDATE public.invoice_reviews
  SET
    status = 'approved',
    reviewer_comment = p_reviewer_comment,
    reviewed_at = now()
  WHERE id = p_review_id
    AND status = 'pending';
END;
$$;


-- 4) Reject: set status='rejected', reviewer_comment, reviewed_at
-- (Only updates if currently pending)
CREATE OR REPLACE FUNCTION public.reject_review(
  p_review_id uuid,
  p_reviewer_comment text
)
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
  UPDATE public.invoice_reviews
  SET
    status = 'rejected',
    reviewer_comment = p_reviewer_comment,
    reviewed_at = now()
  WHERE id = p_review_id
    AND status = 'pending';
END;
$$;

5)

CREATE OR REPLACE FUNCTION public.load_pending_review(p_review_id uuid
)
RETURNS TABLE (
  id uuid,
  image_path text,
  raw_ocr_text text,
  extracted_json jsonb,
  failure_reasons text[],
  status text,
  reviewer_comment text,
  reviewed_at timestamptz,
  created_at timestamptz
)
LANGUAGE sql
SECURITY INVOKER
AS $$
  SELECT
    ir.id,
    ir.image_path,
    ir.raw_ocr_text,
    ir.extracted_json,
    ir.failure_reasons,
    ir.status,
    ir.reviewer_comment,
    ir.reviewed_at,
    ir.created_at
  FROM public.invoice_reviews ir
  WHERE ir.status = 'pending'
  ORDER BY ir.created_at DESC;
$$;


CREATE OR REPLACE FUNCTION public.persist_invoice_batch_v3(p_batch jsonb)
 RETURNS TABLE(out_invoice_ord integer, out_invoice_id integer)
 LANGUAGE plpgsql
 SECURITY DEFINER
AS $function$DECLARE
  v_row record;

  v_invoice_details jsonb;
  v_provider_details jsonb;
  v_patient_details jsonb;

  v_provider_id integer;
  v_patient_id integer;

  v_provider_name text;
  v_provider_address text;
  v_provider_city text;
  v_provider_state text;

  v_patient_first_name text;
  v_patient_last_name text;
  v_dob date;
  v_patient_address text;
  v_patient_city text;
  v_patient_state text;
  v_patient_insurance_policy text;

  v_invoice_number text;
  v_invoice_ord integer;

  v_invoice_date date;
  v_invoice_payment_date date;
  v_invoice_total numeric(18,2);
  v_payment_total numeric(18,2);
  v_invoice_amount_due numeric(9,2);
  v_invoice_insurance_adjustment numeric(9,2);

  v_item jsonb;
  v_line_code text;
  v_line_qty numeric(18,2);
  v_line_amount numeric(18,2);
  v_line_desc text;

  v_new_invoice_id integer;

  v_client_batch_id uuid;
BEGIN
  FOR v_row IN
    SELECT t.value AS value, t.ord AS ord
    FROM jsonb_array_elements(p_batch) WITH ORDINALITY AS t(value, ord)
  LOOP
    v_invoice_details := v_row.value -> 'invoice_details';
    v_provider_details := v_row.value -> 'provider_details';
    v_patient_details := v_row.value -> 'patient_details';

    v_invoice_number := v_invoice_details ->> 'invoice_no';
    v_invoice_ord := COALESCE((v_invoice_details ->> 'invoice_ord')::integer, v_row.ord::integer);

    v_provider_name := v_provider_details ->> 'provider';
    v_provider_address := v_provider_details ->> 'address';
    v_provider_city := v_provider_details ->> 'city';
    v_provider_state := v_provider_details ->> 'state';

    v_client_batch_id := (v_invoice_details ->> 'client_batch_id')::uuid;

    v_patient_first_name := split_part(v_patient_details ->> 'patient', ' ', 1);
    v_patient_last_name := NULLIF(
      trim(
        regexp_replace(
          substr(v_patient_details ->> 'patient', strpos(v_patient_details ->> 'patient', ' ') + 1),
          '\\s+',
          ' ',
          'g'
        )
      ),
      ''
    );

    v_dob := (v_patient_details ->> 'dob')::date;
    v_patient_address := v_patient_details ->> 'address';
    v_patient_city := v_patient_details ->> 'city';
    v_patient_state := v_patient_details ->> 'state';
    v_patient_insurance_policy := v_patient_details ->> 'insurance_policy';

    v_invoice_date := COALESCE(
      NULLIF(v_invoice_details ->> 'invoice_date', '')::date,
      CURRENT_DATE
    );

    v_invoice_payment_date := COALESCE(
      NULLIF(v_invoice_details ->> 'invoice_payment_date', '')::date,
      v_invoice_date
    );

    v_invoice_total := COALESCE(
      NULLIF(v_invoice_details ->> 'invoice_total', '')::numeric,
      0
    );

    v_payment_total := COALESCE(
      NULLIF(v_invoice_details ->> 'payment_total', '')::numeric,
      v_invoice_total
    );

    v_invoice_amount_due := v_invoice_details ->> 'amount_due';
    v_invoice_insurance_adjustment := v_invoice_details ->> 'insurance_adjustment';

    -- Provider upsert
    INSERT INTO public.provider (
      provider_name, provider_address, provider_city, provider_state
    )
    VALUES (
      substring(v_provider_name from 1 for 30),
      substring(v_provider_address from 1 for 35),
      substring(v_provider_city from 1 for 20),
      substring(v_provider_state from 1 for 15)
    )
    ON CONFLICT (provider_name) DO UPDATE
      SET
        provider_address = EXCLUDED.provider_address,
        provider_city = EXCLUDED.provider_city,
        provider_state = EXCLUDED.provider_state
    RETURNING provider_id INTO v_provider_id;

    -- Patient upsert
    INSERT INTO public.patient (
      provider_id, patient_first_name, patient_last_name, dob,
      patient_address, patient_city, patient_state, insurance_policy)
    VALUES (
      v_provider_id,
      v_patient_first_name,
      v_patient_last_name,
      v_dob,
      v_patient_address,
      v_patient_city,
      v_patient_state,
      v_patient_insurance_policy
    )
    ON CONFLICT (provider_id, patient_first_name, patient_last_name, dob) DO UPDATE
      SET
        patient_address = EXCLUDED.patient_address,
        patient_city = EXCLUDED.patient_city,
        patient_state = EXCLUDED.patient_state
    RETURNING patient_id INTO v_patient_id;

    -- Invoice upsert
    INSERT INTO public.invoices (
      invoice_number, patient_id,
      invoice_date, invoice_payment_date, invoice_total, payment_total,
    amount_due, insurance_adjustment)
    VALUES (
      v_invoice_number,
      v_patient_id,
      v_invoice_date,
      v_invoice_payment_date,
      v_invoice_total,
      v_payment_total,
      v_invoice_amount_due,
      v_invoice_insurance_adjustment
    )
    ON CONFLICT (invoice_number) DO UPDATE
      SET
        patient_id = EXCLUDED.patient_id,
        invoice_date = EXCLUDED.invoice_date,
        invoice_payment_date = EXCLUDED.invoice_payment_date,
        invoice_total = EXCLUDED.invoice_total,
        payment_total = EXCLUDED.payment_total
    RETURNING invoice_id INTO v_new_invoice_id;

    -- Invoice batch
    INSERT INTO public.invoice_batch (
      client_batch_id, invoice_ord, invoice_id
    )
    VALUES (
      v_client_batch_id, v_invoice_ord, v_new_invoice_id
    )
    ON CONFLICT (client_batch_id, invoice_ord) DO UPDATE
      SET invoice_id = EXCLUDED.invoice_id;

    -- Ensure invoice_items unique constraint exists
    PERFORM 1
    FROM pg_constraint
    WHERE conname = 'invoice_items_invoice_line_code_key';

    IF NOT FOUND THEN
      BEGIN
        ALTER TABLE public.invoice_items
          ADD CONSTRAINT invoice_items_invoice_line_code_key
          UNIQUE (invoice_id, line_item_code);
      EXCEPTION WHEN duplicate_object THEN
        NULL;
      END;
    END IF;

    -- Invoice items upsert
    FOR v_item IN SELECT * FROM jsonb_array_elements(v_row.value -> 'invoice_line_items')
    LOOP
      v_line_code := v_item ->> 'Code';
      v_line_desc := v_item ->> 'Description';
      v_line_qty := NULLIF((v_item ->> 'Qty')::text, '')::numeric;
      v_line_amount := NULLIF((v_item ->> 'Amount')::text, '')::numeric;

      INSERT INTO public.invoice_items (
        invoice_id, line_item_code, line_item_quantity, line_item_amount, line_item_description
      )
      VALUES (
        v_new_invoice_id, v_line_code, v_line_qty, v_line_amount, v_line_desc
      )
      ON CONFLICT (invoice_id, line_item_code) DO UPDATE
        SET
          line_item_quantity = EXCLUDED.line_item_quantity,
          line_item_amount = EXCLUDED.line_item_amount,
          line_item_description = EXCLUDED.line_item_description;
    END LOOP;

    out_invoice_ord := v_invoice_ord;
    out_invoice_id := v_new_invoice_id;
    RETURN NEXT;
  END LOOP;
END;$function$

CREATE OR REPLACE FUNCTION public.report_invoices_by_date(p_start_date date DEFAULT NULL::date, p_end_date date DEFAULT NULL::date)
 RETURNS TABLE(invoice_date date, invoice_count bigint, sum_subtotal numeric, sum_amount_due numeric, sum_invoice_total numeric)
 LANGUAGE sql
AS $function$
  SELECT
    i.invoice_date,
    COUNT(*) AS invoice_count,
    COALESCE(SUM(i.invoice_total - COALESCE(i.insurance_adjustment, 0)), 0) AS sum_subtotal,
    COALESCE(SUM(i.amount_due), 0) AS sum_amount_due,
    COALESCE(SUM(i.invoice_total), 0) AS sum_invoice_total
  FROM public.invoices i
  WHERE (p_start_date IS NULL OR i.invoice_date >= p_start_date)
    AND (p_end_date IS NULL OR i.invoice_date <= p_end_date)
  GROUP BY i.invoice_date
  ORDER BY invoice_date;
$function$

CREATE OR REPLACE FUNCTION public.report_invoices_by_provider(p_start_date date DEFAULT NULL::date, p_end_date date DEFAULT NULL::date)
 RETURNS TABLE(provider_id integer, provider_name text, invoice_count bigint, sum_subtotal numeric, sum_amount_due numeric, sum_invoice_total numeric)
 LANGUAGE sql
AS $function$
  SELECT
    pr.provider_id,
    pr.provider_name,
    COUNT(*) AS invoice_count,
    COALESCE(SUM(i.invoice_total - COALESCE(i.insurance_adjustment, 0)), 0) AS sum_subtotal,
    COALESCE(SUM(i.amount_due), 0) AS sum_amount_due,
    COALESCE(SUM(i.invoice_total), 0) AS sum_invoice_total
  FROM public.invoices i
  JOIN public.patient pt ON pt.patient_id = i.patient_id
  JOIN public.provider pr ON pr.provider_id = pt.provider_id
  WHERE (p_start_date IS NULL OR i.invoice_date >= p_start_date)
    AND (p_end_date IS NULL OR i.invoice_date <= p_end_date)
  GROUP BY pr.provider_id, pr.provider_name
  ORDER BY sum_amount_due DESC, invoice_count DESC;
$function$

CREATE OR REPLACE FUNCTION public.report_invoices_by_subtotal(p_start_date date DEFAULT NULL::date, p_end_date date DEFAULT NULL::date, p_provider_id integer DEFAULT NULL::integer, p_bucket_count integer DEFAULT 10)
 RETURNS TABLE(bucket_index integer, bucket_low numeric, bucket_high numeric, bucket_label text, invoice_count bigint, sum_subtotal numeric, sum_amount_due numeric)
 LANGUAGE plpgsql
AS $function$
DECLARE
  v_min numeric;
  v_max numeric;
  v_count integer;
  v_width numeric;
BEGIN
  v_count := COALESCE(p_bucket_count, 10);

  SELECT
    MIN((i.invoice_total - COALESCE(i.insurance_adjustment, 0))),
    MAX((i.invoice_total - COALESCE(i.insurance_adjustment, 0)))
  INTO v_min, v_max
  FROM public.invoices i
  JOIN public.patient pt ON pt.patient_id = i.patient_id
  JOIN public.provider pr ON pr.provider_id = pt.provider_id
  WHERE (p_start_date IS NULL OR i.invoice_date >= p_start_date)
    AND (p_end_date IS NULL OR i.invoice_date <= p_end_date)
    AND (p_provider_id IS NULL OR pr.provider_id = p_provider_id);

  IF v_min IS NULL OR v_max IS NULL THEN
    RETURN;
  END IF;

  -- If all values are equal, put everything in a single bucket
  IF v_min = v_max THEN
    bucket_index := 1;
    bucket_low := v_min;
    bucket_high := v_max;
    bucket_label := '$' || to_char(bucket_low, 'FM999999999990D00') || '–' || '$' || to_char(bucket_high, 'FM999999999990D00');

    SELECT
      COUNT(*),
      COALESCE(SUM(i.invoice_total - COALESCE(i.insurance_adjustment, 0)), 0),
      COALESCE(SUM(i.amount_due), 0)
    INTO invoice_count, sum_subtotal, sum_amount_due
    FROM public.invoices i
    JOIN public.patient pt ON pt.patient_id = i.patient_id
    JOIN public.provider pr ON pr.provider_id = pt.provider_id
    WHERE (p_start_date IS NULL OR i.invoice_date >= p_start_date)
      AND (p_end_date IS NULL OR i.invoice_date <= p_end_date)
      AND (p_provider_id IS NULL OR pr.provider_id = p_provider_id)
      AND (i.invoice_total - COALESCE(i.insurance_adjustment, 0)) = v_min;

    RETURN NEXT;
    RETURN;
  END IF;

  v_width := (v_max - v_min) / v_count;

  FOR bucket_index IN 1..v_count LOOP
    bucket_low := v_min + (bucket_index - 1) * v_width;
    bucket_high := v_min + bucket_index * v_width;

    IF bucket_index = v_count THEN
      bucket_high := v_max;
    END IF;

    bucket_label := '$' || to_char(bucket_low, 'FM999999999990D00') || '–' || '$' || to_char(bucket_high, 'FM999999999990D00');

    SELECT
      COUNT(*),
      COALESCE(SUM(i.invoice_total - COALESCE(i.insurance_adjustment, 0)), 0),
      COALESCE(SUM(i.amount_due), 0)
    INTO invoice_count, sum_subtotal, sum_amount_due
    FROM public.invoices i
    JOIN public.patient pt ON pt.patient_id = i.patient_id
    JOIN public.provider pr ON pr.provider_id = pt.provider_id
    WHERE (p_start_date IS NULL OR i.invoice_date >= p_start_date)
      AND (p_end_date IS NULL OR i.invoice_date <= p_end_date)
      AND (p_provider_id IS NULL OR pr.provider_id = p_provider_id)
      AND (i.invoice_total - COALESCE(i.insurance_adjustment, 0)) >= bucket_low
      AND ((i.invoice_total - COALESCE(i.insurance_adjustment, 0)) < bucket_high OR (bucket_index = v_count AND (i.invoice_total - COALESCE(i.insurance_adjustment, 0)) = bucket_high));

    RETURN NEXT;
  END LOOP;
END;
$function$

CREATE OR REPLACE FUNCTION public.report_invoices_by_total_amount_due(p_start_date date DEFAULT NULL::date, p_end_date date DEFAULT NULL::date, p_provider_id integer DEFAULT NULL::integer, p_bucket_count integer DEFAULT 10)
 RETURNS TABLE(bucket_index integer, bucket_low numeric, bucket_high numeric, bucket_label text, invoice_count bigint, sum_amount_due numeric, sum_invoice_total numeric)
 LANGUAGE plpgsql
AS $function$
DECLARE
  v_min numeric;
  v_max numeric;
  v_count integer;
  v_width numeric;
BEGIN
  v_count := COALESCE(p_bucket_count, 10);

  SELECT
    MIN(i.amount_due),
    MAX(i.amount_due)
  INTO v_min, v_max
  FROM public.invoices i
  JOIN public.patient pt ON pt.patient_id = i.patient_id
  JOIN public.provider pr ON pr.provider_id = pt.provider_id
  WHERE (p_start_date IS NULL OR i.invoice_date >= p_start_date)
    AND (p_end_date IS NULL OR i.invoice_date <= p_end_date)
    AND (p_provider_id IS NULL OR pr.provider_id = p_provider_id);

  IF v_min IS NULL OR v_max IS NULL THEN
    RETURN;
  END IF;

  -- If all values are equal, put everything in a single bucket [min, max]
  IF v_min = v_max THEN
    bucket_index := 1;
    bucket_low := v_min;
    bucket_high := v_max;
    bucket_label := '$' || to_char(bucket_low, 'FM999999999990D00') || '–' || '$' || to_char(bucket_high, 'FM999999999990D00');

    SELECT
      COUNT(*),
      COALESCE(SUM(i.amount_due), 0),
      COALESCE(SUM(i.invoice_total), 0)
    INTO invoice_count, sum_amount_due, sum_invoice_total
    FROM public.invoices i
    JOIN public.patient pt ON pt.patient_id = i.patient_id
    JOIN public.provider pr ON pr.provider_id = pt.provider_id
    WHERE (p_start_date IS NULL OR i.invoice_date >= p_start_date)
      AND (p_end_date IS NULL OR i.invoice_date <= p_end_date)
      AND (p_provider_id IS NULL OR pr.provider_id = p_provider_id)
      AND i.amount_due = v_min;

    RETURN NEXT;
    RETURN;
  END IF;

  v_width := (v_max - v_min) / v_count;

  FOR bucket_index IN 1..v_count LOOP
    bucket_low := v_min + (bucket_index - 1) * v_width;
    bucket_high := v_min + bucket_index * v_width;

    -- Ensure last bucket reaches v_max exactly (avoid rounding gaps)
    IF bucket_index = v_count THEN
      bucket_high := v_max;
    END IF;

    bucket_label := '$' || to_char(bucket_low, 'FM999999999990D00') || '–' || '$' || to_char(bucket_high, 'FM999999999990D00');

    SELECT
      COUNT(*),
      COALESCE(SUM(i.amount_due), 0),
      COALESCE(SUM(i.invoice_total), 0)
    INTO invoice_count, sum_amount_due, sum_invoice_total
    FROM public.invoices i
    JOIN public.patient pt ON pt.patient_id = i.patient_id
    JOIN public.provider pr ON pr.provider_id = pt.provider_id
    WHERE (p_start_date IS NULL OR i.invoice_date >= p_start_date)
      AND (p_end_date IS NULL OR i.invoice_date <= p_end_date)
      AND (p_provider_id IS NULL OR pr.provider_id = p_provider_id)
      AND i.amount_due >= bucket_low
      AND (i.amount_due < bucket_high OR (bucket_index = v_count AND i.amount_due = bucket_high));

    RETURN NEXT;
  END LOOP;
END;
$function$

