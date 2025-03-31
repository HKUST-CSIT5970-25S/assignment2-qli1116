package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Partitioner;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

/**
 * Compute bigram relative frequencies using the "pairs" approach.
 * 
 * For each left word A, the program outputs:
 *   1. A marginal count record with key (A, "") and value totalCount (e.g., 2.0)
 *   2. One record for each bigram (A, B) with value equal to P(B|A)
 * 
 * The output is tab-delimited. For example:
 * 
 * It        2.0
 * It    was 1.0
 * best      1.0
 * best  of  1.0
 * ...
 */
public class BigramFrequencyPairs extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyPairs.class);

    /**
     * Mapper: for each pair of consecutive words, emit:
     *   (A, B) with count 1, and
     *   (A, "*") with count 1 for the marginal count.
     */
    private static class MyMapper extends Mapper<LongWritable, Text, PairOfStrings, IntWritable> {
        private static final IntWritable ONE = new IntWritable(1);
        private static final PairOfStrings BIGRAM = new PairOfStrings();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] words = line.trim().split("\\s+");
            if (words.length < 2)
                return;

            for (int i = 0; i < words.length - 1; i++) {
                if (words[i].isEmpty() || words[i+1].isEmpty())
                    continue;

                // Emit the bigram (A, B)
                BIGRAM.set(words[i], words[i+1]);
                context.write(BIGRAM, ONE);
                // Emit the marginal count (A, "*")
                BIGRAM.set(words[i], "*");
                context.write(BIGRAM, ONE);
            }
        }
    }

    /**
     * Combiner: sums counts for each key.
     */
    private static class MyCombiner extends Reducer<PairOfStrings, IntWritable, PairOfStrings, IntWritable> {
        private static final IntWritable SUM = new IntWritable();

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            SUM.set(sum);
            context.write(key, SUM);
        }
    }

    /**
     * Partitioner: partition by the left element of the pair.
     */
    private static class MyPartitioner extends Partitioner<PairOfStrings, IntWritable> {
        @Override
        public int getPartition(PairOfStrings key, IntWritable value, int numReduceTasks) {
            return (key.getLeftElement().hashCode() & Integer.MAX_VALUE) % numReduceTasks;
        }
    }

    /**
     * Reducer: For each left word A, output first the marginal count (total count)
     * and then the relative frequencies for each succeeding word.
     * 
     * Assumes that the key (A, "*") (the marginal record) is processed before any key (A, B)
     * for a given left word.
     */
    private static class MyReducer extends Reducer<PairOfStrings, IntWritable, PairOfStrings, FloatWritable> {
        private final static FloatWritable RESULT = new FloatWritable();
        private String currentLeft = null;
        private int marginalCount = 0;

        @Override
        public void reduce(PairOfStrings key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {
            String left = key.getLeftElement();
            String right = key.getRightElement();
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }

            // When a new left word is encountered, reset the marginal count.
            if (currentLeft == null || !currentLeft.equals(left)) {
                currentLeft = left;
                marginalCount = 0;
            }

            if (right.equals("*")) {
                // This record holds the marginal count.
                // Emit a record with key (A, "") to represent the total count.
                PairOfStrings outKey = new PairOfStrings(left, "");
                RESULT.set((float) sum);
                context.write(outKey, RESULT);
                marginalCount = sum;
            } else {
                // For a regular bigram, output relative frequency if marginal count is available.
                if (marginalCount > 0) {
                    float relativeFrequency = (float) sum / marginalCount;
                    RESULT.set(relativeFrequency);
                    context.write(key, RESULT);
                }
            }
        }
    }

    public BigramFrequencyPairs() { }

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @Override
    public int run(String[] args) throws Exception {
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg()
                .withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg()
                .withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLineParser parser = new GnuParser();
        CommandLine cmdline;
        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: " + exp.getMessage());
            return -1;
        }

        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            System.out.println("args: " + Arrays.toString(args));
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        LOG.info("Tool: " + BigramFrequencyPairs.class.getSimpleName());
        LOG.info(" - input path: " + inputPath);
        LOG.info(" - output path: " + outputPath);
        LOG.info(" - number of reducers: " + reduceTasks);

        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyPairs.class.getSimpleName());
        job.setJarByClass(BigramFrequencyPairs.class);
        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(PairOfStrings.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setPartitionerClass(MyPartitioner.class);
        job.setReducerClass(MyReducer.class);

        Path outputDir = new Path(outputPath);
        FileSystem.get(conf).delete(outputDir, true);

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyPairs(), args);
    }
}
