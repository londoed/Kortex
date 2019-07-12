module Kortex {

  use Sort, BigInteger, Math, Random;
  /*
    Matrix Operations from NumPy.
  */

  proc random_choice(a: [] ?t, size: int=1, replace: bool=true) {
    var b: [1..0] a.eltType,
        result: [1..0] a.eltType;

    for i in a {
      b.append(a);
    }

    if !replace {
      shuffle(b);
      result = b[1..size];
    } else {
      for i in 1..size {
        shuffle(b);
        result.append(b[1]);
        b.remove(1);
      }
    }
    return result;
  }

  proc random_choice(a: [] ?t, size: int=1, replace: bool=true, p: [] ?u) {
    return choose_multinomial(a=a, size=size, replace=replace, p=p);
  }

  proc choose_multinomial(a: [] ?t, size: int, replace: bool=true, p: [] real) {
    var result: [1..0] a.eltType,
        r: [1..size] real,
        b = for i in a do i:a.eltType,
        q = for i in p do i:p.eltType;

    fill_random(r);

    for i in 1..size {
      var sum: real = 0,
          k: int = 1,
          denom: real = + reduce q;
      const c = r[i];

      do {
        sum += q[k] / denom;
        k += 1;
      } while sum <= c;
      result.append(b[k - 1]);
      
      if !replace {
        b.remove(k - 1);
        q.remove(k - 1);
      }
    }
    return result;
  }
  }
}
