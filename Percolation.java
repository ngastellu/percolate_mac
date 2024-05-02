import java.lang.reflect.Array;

public class Percolation {
    static int size = 10;
    static int length = size*size;
    static boolean[] grid = new boolean[length];

    public static void init(){
        UnionFind.init(length+2);
        for (int i=0; i<length; i++) {
            grid[i] = false;
        }
    } 

    public static void print() {
        String line = "";
        for (int k=0; k<length; k++) {
            if (grid[k]) {
                line = line+"*";
            }
            else {
                line = line+"_";
            }
            if ((k+1)%size == 0) {
                System.out.println(line);
                line = "";
            }
        }
    }

    public static int randomShadow() {
        int index = -1;
        while (index == -1) {
            int potential_index = (int) Math.floor(length*Math.random());
            if (!grid[potential_index]) {
                index = potential_index;
            }
        }
        grid[index] = true;
        propagateUnion(index);
        return index;
    }

    public static boolean detectPath(boolean[] seen, int n, boolean up) {
        // We only care about the case where the tile we're on is black
        if (grid[n]&&(!seen[n])) {
            seen[n] = true;
            // Base cases
            if ((n<size)&&(up)) {
                return true;
            }
            if ((length-size <= n)&&(!up)) {
                return true;
            }
            // We check to the right
            if ((n+1)%size != 0) {
                boolean next = detectPath(seen, n+1, up);
                if (next){
                    return next;
                }
            }
            // We check to the left
            if (n%size != 0) {
                boolean next = detectPath(seen, n-1, up);
                if (next){
                    return next;
                }
            }
            // We check above
            if (n>=size) {
                boolean next = detectPath(seen, n-size, up);
                if (next){
                    return next;
                }
            }
            // We check below
            if (n+size<length) {
                boolean next = detectPath(seen, n+size, up);
                if (next){
                    return next;
                }
            }
            return false;
        }
        else {
            return false;
        }
    }

    public static boolean isNaivePercolation(int n) {
        boolean[] seen_up = new boolean[length];
        boolean[] seen_down = new boolean[length];
        return detectPath(seen_down, n, false)&&detectPath(seen_up, n, true);
    }

    public static boolean isFastPercolation(int n){
        int n_class = UnionFind.find(n);
        boolean flag_up = false;
        boolean flag_down = false;
        //We see if there as top liner and a bottom liner in the class of n
        for (int i = 0; i < size; i++) {
            flag_up = flag_up || (UnionFind.find(i)==n_class);
            flag_down = flag_down || (UnionFind.find(length-(i+1))==n_class);
            if (flag_up && flag_down) return true;
        }
        return false;
    }

    public static boolean isLogPercolation(){
        return (UnionFind.find(0) == UnionFind.find(length+1));
    }

    public static boolean isPercolation(int n) {
        return isLogPercolation();
    }

    public static double percolation() {
        init();
        int k;
        boolean percolated = false;
        int black_cell_count = 0;
        while (!percolated) {
            k = randomShadow();
            black_cell_count += 1;
            percolated = isPercolation(k);
                
        }
        return ((double) black_cell_count)/length;
    }

    public static double monteCarlo(int n){
        double sum = 0;
        for (int i = 0; i < n; i++) {
            sum += percolation();
        }
        return (sum/n);
    }

    public static void propagateUnion(int x) {
        // We check to the right
        if (((x+1)%size != 0)&&(grid[x+1])) {
            UnionFind.union(x+2, x+1);
        }
        // We check to the left
        if ((x%size != 0)&&(grid[x-1])) {
            UnionFind.union(x, x+1);
        }
        // We check above
        if ((x>=size)&&(grid[x-size])) {
            UnionFind.union(x-size+1, x+1);
        }
        else if (x<size) {
            UnionFind.union(0,x+1);
        }
        // We check below
        if ((x+size<length)&&(grid[x+size])) {
            UnionFind.union(x+size+1, x+1);
        }
        else if (x+size>=length) {
            UnionFind.union(length+1, x+1);
        }
    }

    public static void main(String[] args) {
        int n = Integer.parseInt(args[0]);
        long start = System.currentTimeMillis();
        double threshold = monteCarlo(n);
        long end = System.currentTimeMillis();
        System.out.println(threshold);
        System.out.println(end-start);
    }
}

